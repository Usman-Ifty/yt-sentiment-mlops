document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const statusEl = document.getElementById('status');
    const statsContainer = document.getElementById('statsContainer');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const summaryEl = document.getElementById('summary');
    const cloudEl = document.getElementById('cloudTags');
    const rankEl = document.getElementById('vibeRank');
    
    analyzeBtn.disabled = true;
    analyzeBtn.style.opacity = '0.5';
    statusEl.innerText = "Extracting data...";
    statsContainer.style.display = 'none';

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

            // Audit & Rank Logic
            const health = Math.max(0, posPct - (negPct * 1.5));
            let rank = 'F';
            let rankColor = '#ff3e3e';

            if (health > 80) { rank = 'S'; rankColor = '#00ff9d'; }
            else if (health > 65) { rank = 'A'; rankColor = '#00ff9d'; }
            else if (health > 50) { rank = 'B'; rankColor = '#f7b731'; }
            else if (health > 35) { rank = 'C'; rankColor = '#f7b731'; }
            else if (health > 20) { rank = 'D'; rankColor = '#ff3e3e'; }

            rankEl.innerText = rank;
            rankEl.style.background = rankColor;

            // Top 10 Keywords
            const topWords = getTopTenWords(comments);

            document.getElementById('posCount').innerText = posPct + '%';
            document.getElementById('neuCount').innerText = neuPct + '%';
            document.getElementById('negCount').innerText = negPct + '%';
            
            document.getElementById('barPos').style.width = posPct + '%';
            document.getElementById('barNeu').style.width = neuPct + '%';
            document.getElementById('barNeg').style.width = negPct + '%';

            cloudEl.innerHTML = "";
            topWords.forEach(w => {
                const span = document.createElement('span');
                span.className = 'tag';
                span.innerText = w.toUpperCase();
                cloudEl.appendChild(span);
            });

            if (rank === 'S' || rank === 'A') {
                summaryEl.innerText = "Audience Sentiment: Excellent";
                summaryEl.style.color = "#00ff9d";
            } else if (rank === 'B' || rank === 'C') {
                summaryEl.innerText = "Audience Sentiment: Mixed";
                summaryEl.style.color = "#f7b731";
            } else {
                summaryEl.innerText = "Audience Sentiment: Negative";
                summaryEl.style.color = "#ff3e3e";
            }

            statsContainer.style.display = 'block';
            statusEl.innerText = "Analysis Complete";

            chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: paintComments,
                args: [predictions]
            });

        } catch (err) {
            statusEl.innerText = "Communication offline.";
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.style.opacity = '1';
            analyzeBtn.innerText = "RE-RUN SENTIMENT AUDIT";
        }
    });
});

function getTopTenWords(comments) {
    const stops = [
        'the','is','and','to','a','in','it','i','this','that','of','for','on','u','my','was','not','be',
        'with','are','if','so','but','at','as','or','by','have','video','your','you'
    ];
    let words = {};
    comments.forEach(c => {
        c.toLowerCase().split(/\s+/).forEach(w => {
            const clean = w.replace(/[^a-z]/g, '');
            if(clean.length > 3 && !stops.includes(clean)) {
                words[clean] = (words[clean] || 0) + 1;
            }
        });
    });
    let sorted = Object.keys(words).sort((a,b) => words[b] - words[a]);
    return sorted.slice(0, 10);
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

function paintComments(predictions) {
    const commentNodes = document.querySelectorAll('ytd-comment-thread-renderer');
    let validIndex = 0;
    for (let i = 0; i < commentNodes.length; i++) {
        if (validIndex >= predictions.length) break;
        const textNode = commentNodes[i].querySelector('#content-text');
        if (!textNode || !textNode.innerText.trim()) continue;

        const pred = predictions[validIndex];
        const mainCard = commentNodes[i].querySelector('#body');
        if (mainCard) {
            mainCard.style.borderLeft = `8px solid ${pred.label==='positive' ? '#00ff9d' : pred.label==='negative' ? '#ff3e3e' : '#f7b731'}`;
            mainCard.style.backgroundColor = "rgba(255,255,255,0.02)";
            mainCard.style.paddingLeft = "20px";
            mainCard.style.transition = "0.5s";
            
            const nameEl = mainCard.querySelector('#header-author');
            if (nameEl && !mainCard.querySelector('.ifty-tag')) {
                const tag = document.createElement('span');
                tag.className = 'ifty-tag';
                tag.style.fontSize = "10px"; tag.style.fontWeight = "900"; tag.style.marginLeft = "12px";
                tag.style.color = pred.label==='positive' ? '#00ff9d' : pred.label==='negative' ? '#ff3e3e' : '#f7b731';
                tag.innerText = pred.label.toUpperCase();
                nameEl.appendChild(tag);
            }
        }
        validIndex++;
    }
}
