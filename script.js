document.getElementById('loanForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = {
        name: document.getElementById('name').value,
        income: document.getElementById('income').value,
        loanAmount: document.getElementById('loanAmount').value,
        creditScore: document.getElementById('creditScore').value,
        married: document.getElementById('married').value,
        education: document.getElementById('education').value,
        propertyArea: document.getElementById('propertyArea').value,
    };

    try {
        const response = await fetch('http://localhost:8000/predict/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData),
        });

        const result = await response.json();
        const resultEl = document.getElementById('result');
        const nameEl = resultEl.querySelector('.name');
        const badgeEl = resultEl.querySelector('.badge');
        const msgEl = resultEl.querySelector('.message');

        // Reset
        resultEl.hidden = false;
        resultEl.className = 'result';
        nameEl.textContent = formData.name || 'Applicant';
        msgEl.textContent = '';

        if (result.approval) {
            const approvalText = result.approval.toString().toLowerCase();
            badgeEl.textContent = result.approval.toString();
            
            // Check if rejected
            if (approvalText.includes('rejected') || approvalText === 'no') {
                resultEl.classList.add('rejected');
                msgEl.textContent = 'This application does not meet the current approval criteria. Consider improving credit score or income.';
            } else {
                resultEl.classList.add('success');
                msgEl.textContent = 'This is a model-based prediction. Contact the lender for final decision.';
            }
        } else if (result.error) {
            badgeEl.textContent = 'Error';
            resultEl.classList.add('error');
            msgEl.textContent = result.error || 'Unexpected error from server.';
        } else {
            badgeEl.textContent = 'Unknown';
            resultEl.classList.add('muted');
            msgEl.textContent = 'No prediction available.';
        }
    } catch (err) {
        const resultEl = document.getElementById('result');
        const nameEl = resultEl.querySelector('.name');
        const badgeEl = resultEl.querySelector('.badge');
        const msgEl = resultEl.querySelector('.message');

        resultEl.hidden = false;
        resultEl.className = 'result error';
        nameEl.textContent = formData.name || 'Applicant';
        badgeEl.textContent = 'Error';
        msgEl.textContent = 'Could not connect to server.';
    }
});
