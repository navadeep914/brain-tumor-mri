const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('scanPreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const fileName = document.getElementById('fileName');
const scanState = document.getElementById('scanState');
const resultEmpty = document.getElementById('resultEmpty');
const resultContent = document.getElementById('resultContent');
const resultBadge = document.getElementById('resultBadge');
const tumorStatus = document.getElementById('tumorStatus');
const themeToggle = document.getElementById('themeToggle');
let selectedFile = null;
let latestResult = null;

const featureCard = document.createElement('article');
featureCard.className = 'output-card';
featureCard.innerHTML = '<div class="output-card-head"><b>4. ICA feature map</b><span>Feature reconstruction</span></div><div class="output-image" id="featureMapOutput"><div class="output-placeholder">Available after analysis</div></div><p>Image reconstructed from the independent components used by the classifier.</p>';
document.querySelector('.output-grid').appendChild(featureCard);

fileInput.addEventListener('change', event => {
	selectedFile = event.target.files[0];
	if (!selectedFile) return;
	if (selectedFile.size > 10 * 1024 * 1024) {
		fileName.textContent = 'File exceeds 10 MB';
		return;
	}
	fileName.textContent = selectedFile.name;
	scanState.textContent = 'Ready';
	analyzeBtn.disabled = false;
	const reader = new FileReader();
	reader.onload = loadEvent => {
		preview.style.backgroundImage = `url(${loadEvent.target.result})`;
		preview.classList.add('has-image');
		setOutput('originalOutput', loadEvent.target.result);
	};
	reader.readAsDataURL(selectedFile);
});

function setOutput(id, source) {
	const element = document.getElementById(id);
	element.innerHTML = '';
	const image = document.createElement('img');
	image.src = source;
	image.alt = `${id.replace('Output', '')} MRI output`;
	element.appendChild(image);
}

function updateProbabilities(probabilities) {
	document.querySelectorAll('.prob-row').forEach(row => {
		const label = row.querySelector('span').textContent;
		const key = label === 'No tumor' ? 'notumor' : label.toLowerCase();
		const value = (probabilities?.[key] || 0) * 100;
		row.querySelector('i b').style.width = `${value}%`;
		row.querySelector('strong').textContent = `${value.toFixed(1)}%`;
	});
}

analyzeBtn.addEventListener('click', async () => {
	if (!selectedFile) return;
	analyzeBtn.disabled = true;
	analyzeBtn.innerHTML = 'Analyzing <span>…</span>';
	scanState.textContent = 'Processing';
	try {
		const form = new FormData();
		form.append('file', selectedFile);
		const response = await fetch('http://localhost:5000/analyze', { method: 'POST', body: form });
		if (!response.ok) throw new Error('API unavailable');
		latestResult = await response.json();
		const result = latestResult;
		resultEmpty.hidden = true;
		resultContent.hidden = false;
		const isTumorPositive = Boolean(result.tumor_detected);
		resultBadge.textContent = isTumorPositive ? 'Tumor positive' : 'Tumor negative';
		resultBadge.style.background = isTumorPositive ? '#fef2f2' : '#ecfdf5';
		resultBadge.style.color = isTumorPositive ? '#b91c1c' : '#047857';
		document.getElementById('resultLabel').textContent = result.label === 'notumor'
			? 'No tumor'
			: result.label[0].toUpperCase() + result.label.slice(1);
		document.getElementById('resultStatus').textContent = isTumorPositive
			? 'Tumor positive · review visual aids'
			: 'Tumor negative';
		tumorStatus.innerHTML = isTumorPositive
			? '<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#dc2626;"></span> Tumor status: POSITIVE'
			: '<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#22c55e;"></span> Tumor status: NEGATIVE';
		document.getElementById('confidenceValue').textContent = `${(result.confidence * 100).toFixed(1)}%`;
		document.getElementById('confidenceBar').style.width = `${result.confidence * 100}%`;
		updateProbabilities(result.probabilities);
		if (result.images) {
			setOutput('originalOutput', result.images.original);
			setOutput('highlightOutput', result.images.highlight);
			setOutput('segmentationOutput', result.images.segmentation);
			setOutput('featureMapOutput', result.images.feature_map);
		}
		scanState.textContent = 'Complete';
	} catch (error) {
		resultEmpty.hidden = false;
		resultContent.hidden = true;
		resultEmpty.querySelector('b').textContent = 'Analysis service unavailable';
		resultEmpty.querySelector('p').textContent = 'Start the Python API on port 5000 to preserve the original model output.';
		if (tumorStatus) {
			tumorStatus.innerHTML = '<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#d1d5db;"></span> Tumor status: —';
		}
		scanState.textContent = 'API offline';
	} finally {
		analyzeBtn.disabled = false;
		analyzeBtn.innerHTML = 'Run analysis <span>→</span>';
	}
});

themeToggle.addEventListener('click', () => {
	const html = document.documentElement;
	const dark = html.dataset.theme === 'dark';
	html.dataset.theme = dark ? 'light' : 'dark';
	themeToggle.textContent = dark ? '◐' : '☼';
});

function loadPdfLibrary() {
	if (window.jspdf?.jsPDF) return Promise.resolve();
	return new Promise((resolve, reject) => {
		const script = document.createElement('script');
		script.src = '/node_modules/jspdf/dist/jspdf.umd.min.js';
		script.onload = resolve;
		script.onerror = () => reject(new Error('Could not load the PDF generator.'));
		document.head.appendChild(script);
	});
}

document.getElementById('downloadBtn').addEventListener('click', async event => {
	if (!latestResult) return;
	const button = event.currentTarget;
	button.disabled = true;
	button.textContent = 'Preparing report...';
	try {
		await loadPdfLibrary();
		const { jsPDF } = window.jspdf;
		const report = new jsPDF();
		const result = latestResult;
		const label = result.label === 'notumor' ? 'No tumor' : result.label;
		const status = result.tumor_detected ? 'Tumor status: POSITIVE' : 'Tumor status: NEGATIVE';

		report.setFontSize(20);
		report.text('NeuroLens MRI Analysis Report', 14, 20);
		report.setFontSize(11);
		report.text(`Scan: ${selectedFile?.name || 'MRI image'}`, 14, 34);
		report.text(`Predicted class: ${label}`, 14, 43);
		report.text(`Confidence: ${(result.confidence * 100).toFixed(1)}%`, 14, 52);
		report.text(`Status: ${status}`, 14, 61);
		report.text('Method: ICA feature extraction + Kernel SVM', 14, 70);
		report.setFontSize(13);
		report.text('Class probabilities', 14, 84);
		report.setFontSize(10);
		const classNames = { glioma: 'Glioma', meningioma: 'Meningioma', notumor: 'No tumor', pituitary: 'Pituitary' };
		Object.entries(result.probabilities || {}).forEach(([key, value], index) => {
			report.text(`${classNames[key] || key}: ${(value * 100).toFixed(1)}%`, 14, 94 + index * 7);
		});
		report.setFontSize(9);
		report.text(
			'Visual outputs are review aids, not a substitute for clinical interpretation. The highlight and segmentation are heuristic; the feature map shows the ICA reconstruction.',
			14,
			132,
			{ maxWidth: 182 },
		);

		report.addPage();
		report.setFontSize(16);
		report.text('Model visual outputs', 14, 20);
		const visuals = [
			['Original MRI', result.images.original],
			['Tumor highlighting', result.images.highlight],
			['Segmentation mask', result.images.segmentation],
			['ICA feature map', result.images.feature_map],
		];
		visuals.forEach(([title, image], index) => {
			const x = 14 + (index % 2) * 94;
			const y = 31 + Math.floor(index / 2) * 124;
			report.setFontSize(10);
			report.text(title, x, y);
			if (image) report.addImage(image, 'PNG', x, y + 4, 82, 82, undefined, 'FAST');
		});
		report.save('neurolens-mri-report.pdf');
	} catch (error) {
		alert(error.message || 'Could not create the PDF report.');
	} finally {
		button.disabled = false;
		button.textContent = '↓ Download PDF report';
	}
});
