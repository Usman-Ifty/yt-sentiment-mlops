document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const statusEl = document.getElementById('status');
    const statsContainer = document.getElementById('statsContainer');
    const emojiContainer = document.getElementById('emojiContainer');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const summaryEl = document.getElementById('summary');
    const cloudEl = document.getElementById('cloudTags');
    const emojiCloudEl = document.getElementById('emojiCloud');
    const rankEl = document.getElementById('vibeRank');
    
    analyzeBtn.disabled = true;
    analyzeBtn.style.opacity = '0.5';
    statusEl.innerText = "Extracting data...";
    statsContainer.style.display = 'none';
    emojiContainer.style.display = 'none';

    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab.url.includes("youtube.com/watch")) {
        statusEl.innerText = "Please open a YouTube video.";
        analyzeBtn.disabled = false;
        analyzeBtn.style.opacity = '1';
        return;
    }

    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: scrapeComments
    }, async (injectionResults) => {
        if (!injectionResults || !injectionResults[0]) {
            statusEl.innerText = "Process Failed.";
            analyzeBtn.disabled = false;
            return;
        }

        let comments = injectionResults[0].result;
        if (comments.length === 0) {
            statusEl.innerText = "No comments found. Scroll down!";
            analyzeBtn.disabled = false;
            analyzeBtn.style.opacity = '1';
            return;
        }

        statusEl.innerText = `Analyzing ${comments.length} comments...`;

        try {
            const response = await fetch("http://16.170.230.153:5000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ texts: comments })
            });
            
            if (!response.ok) throw new Error("AWS Offline");
            
            const data = await response.json();
            const predictions = data.predictions;
            
            let pos=0, neu=0, neg=0;
            predictions.forEach(p => {
                if(p.label === 'positive') pos++;
                else if(p.label === 'neutral') neu++;
                else if(p.label === 'negative') neg++;
            });

            const total = predictions.length;
            const posPct = Math.round((pos/total)*100);
            const neuPct = Math.round((neu/total)*100);
            const negPct = Math.round((neg/total)*100);

            const health = Math.max(0, posPct - (negPct * 1.3));
            let rank = 'F';
            let rankColor = '#ff3e3e';

            if (health > 80) { rank = 'S'; rankColor = '#00ff9d'; }
            else if (health > 65) { rank = 'A'; rankColor = '#00ff9d'; }
            else if (health > 50) { rank = 'B'; rankColor = '#f7b731'; }
            else if (health > 35) { rank = 'C'; rankColor = '#f7b731'; }
            else if (health > 20) { rank = 'D'; rankColor = '#ff3e3e'; }

            rankEl.innerText = rank;
            rankEl.style.background = rankColor;

            // Stats
            document.getElementById('posCount').innerText = posPct + '%';
            document.getElementById('neuCount').innerText = neuPct + '%';
            document.getElementById('negCount').innerText = negPct + '%';
            
            document.getElementById('barPos').style.width = posPct + '%';
            document.getElementById('barNeu').style.width = neuPct + '%';
            document.getElementById('barNeg').style.width = negPct + '%';

            // Keyword Cloud
            cloudEl.innerHTML = "";
            getTopTenWords(comments).forEach(w => {
                const span = document.createElement('span');
                span.className = 'tag';
                span.innerText = w.toUpperCase();
                cloudEl.appendChild(span);
            });

            // Emoji Cloud
            emojiCloudEl.innerHTML = "";
            getTopEmojis(comments).forEach(e => {
                const span = document.createElement('span');
                span.innerText = e;
                emojiCloudEl.appendChild(span);
            });

            if (rank === 'S' || rank === 'A') {
                summaryEl.innerText = "Overall Sentiment: Excellent";
                summaryEl.style.color = "#00ff9d";
            } else if (rank === 'B' || rank === 'C') {
                summaryEl.innerText = "Overall Sentiment: Mixed";
                summaryEl.style.color = "#f7b731";
            } else {
                summaryEl.innerText = "Overall Sentiment: Negative";
                summaryEl.style.color = "#ff3e3e";
            }

            statsContainer.style.display = 'block';
            emojiContainer.style.display = 'block';
            statusEl.innerText = "Analysis Complete";

            chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: paintComments,
                args: [predictions, rank, rankColor]
            });

        } catch (err) {
            statusEl.innerText = "Communication offline.";
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.style.opacity = '1';
            analyzeBtn.innerText = "RE-SCAN SENTIMENT";
        }
    });
});

function getTopEmojis(comments) {
    const emojiRegex = /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu;
    let counts = {};
    comments.forEach(c => {
        const matches = c.match(emojiRegex);
        if (matches) {
            matches.forEach(e => counts[e] = (counts[e] || 0) + 1);
        }
    });
    return Object.keys(counts).sort((a,b) => counts[b] - counts[a]).slice(0, 6);
}

function getTopTenWords(comments) {
    const stops = ['the','is','and','to','a','in','it','i','this','that','of','for','on','u','my','was','not','be','with','are','if','so','but','at','as','or','by','have','video','your','you', 'very', 'they'];
    let words = {};
    comments.forEach(c => {
        c.toLowerCase().split(/\s+/).forEach(w => {
            const clean = w.replace(/[^a-z]/g, '');
            if(clean.length > 3 && !stops.includes(clean)) {
                words[clean] = (words[clean] || 0) + 1;
            }
        });
    });
    return Object.keys(words).sort((a,b) => words[b] - words[a]).slice(0, 10);
}

function scrapeComments() {
    const commentNodes = document.querySelectorAll('ytd-comment-thread-renderer #content-text');
    let texts = [];
    for (let i = 0; i < commentNodes.length; i++) {
        const text = commentNodes[i].innerText;
        if (text && text.trim().length > 0) texts.push(text);
    }
    return texts;
}

function paintComments(predictions, rank, rankColor) {
    const commentNodes = document.querySelectorAll('ytd-comment-thread-renderer');
    
    // 1. Inject Vibe Ticker at top of comment section
    const commentSection = document.querySelector('ytd-comments');
    if (commentSection && !document.getElementById('iftyVibeTicker')) {
        const ticker = document.createElement('div');
        ticker.id = 'iftyVibeTicker';
        ticker.style.padding = '15px'; ticker.style.margin = '10px 0';
        ticker.style.background = 'rgba(0,0,0,0.85)'; ticker.style.backdropFilter = 'blur(10px)';
        ticker.style.border = `2px solid ${rankColor}`; ticker.style.borderRadius = '12px';
        ticker.style.color = '#fff'; ticker.style.fontFamily = 'Inter, sans-serif';
        ticker.style.display = 'flex'; ticker.style.justifyContent = 'space-between'; ticker.style.alignItems = 'center';
        ticker.innerHTML = `
            <div style="font-weight: 900; font-size: 14px; letter-spacing: 1px;">CHANNEL SENTIMENT VIBE: <span style="font-size: 24px; color: ${rankColor}; margin-left: 10px;">${rank} GRADE</span></div>
            <div style="font-size: 10px; color: #888; font-weight: 500;">SECURED BY IFTY & MASHOOD AI</div>
        `;
        const itemSection = commentSection.querySelector('#contents');
        if (itemSection) itemSection.prepend(ticker);
    } else if (document.getElementById('iftyVibeTicker')) {
        document.getElementById('iftyVibeTicker').style.borderColor = rankColor;
        document.getElementById('iftyVibeTicker').querySelector('span').innerText = `${rank} GRADE`;
        document.getElementById('iftyVibeTicker').querySelector('span').style.color = rankColor;
    }

    // 2. Add AI reply buttons & tags to comments
    let validIndex = 0;
    for (let i = 0; i < commentNodes.length; i++) {
        if (validIndex >= predictions.length) break;
        const textNode = commentNodes[i].querySelector('#content-text');
        if (!textNode || !textNode.innerText.trim()) continue;

        const pred = predictions[validIndex];
        const mainCard = commentNodes[i].querySelector('#body');
        if (mainCard) {
            mainCard.style.borderLeft = `6px solid ${pred.label==='positive' ? '#00ff9d' : pred.label==='negative' ? '#ff3e3e' : '#f7b731'}`;
            mainCard.style.backgroundColor = "rgba(255,255,255,0.01)";
            
            const header = mainCard.querySelector('#header-author');
            if (header && !header.querySelector('.ifty-tag')) {
                const tag = document.createElement('span');
                tag.className = 'ifty-tag';
                tag.style.fontSize = "10px"; tag.style.fontWeight = "900"; tag.style.marginLeft = "12px";
                tag.style.color = pred.label==='positive' ? '#00ff9d' : pred.label==='negative' ? '#ff3e3e' : '#f7b731';
                tag.innerText = pred.label.toUpperCase();
                header.appendChild(tag);

                // AI Suggested Reply
                const replySuggest = document.createElement('span');
                replySuggest.style.fontSize = "10px"; replySuggest.style.marginLeft = "12px"; replySuggest.style.color = "#888"; 
                replySuggest.style.cursor = "pointer"; replySuggest.style.fontStyle = "italic";
                replySuggest.innerText = "💡 Suggest Reply";
                replySuggest.onclick = () => {
                   const suggestions = {
                      'positive': ["W video! fr", "Amazing content as always! 💪", "Slayed this one! 🔥"],
                      'negative': ["Why so pressed? 💀", "L take fr", "Touched grass recently? 🌳"],
                      'neutral': ["Solid perspective.", "Interesting point! 💯", "That's a mood."]
                   };
                   const list = suggestions[pred.label];
                   const random = list[Math.floor(Math.random()*list.length)];
                   alert(`IFTY AI Suggestion: "${random}"`);
                };
                header.appendChild(replySuggest);
            }
        }
        validIndex++;
    }
}

