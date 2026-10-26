const API_BASE_URL = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:')
	? 'http://localhost:5000'
	: 'https://brain-tumor-mri-46od.onrender.com';

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

const modelSelect = document.getElementById('modelSelect');
if (modelSelect) {
	modelSelect.addEventListener('change', () => {
		if (selectedFile && !analyzeBtn.disabled) {
			analyzeBtn.click();
		}
	});
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

function updateKsvmDiagnostics(ksvm, predictedLabel) {
	if (!ksvm) return;
	const decisionScores = ksvm.decision_scores || {};
	const activeMargin = decisionScores[predictedLabel] !== undefined ? decisionScores[predictedLabel] : 0;
	const marginSign = activeMargin >= 0 ? '+' : '';

	// Top badges
	const chipKernel = document.getElementById('chipKernel');
	const chipMargin = document.getElementById('chipMargin');
	const chipVectors = document.getElementById('chipVectors');
	if (chipKernel) chipKernel.textContent = `${ksvm.kernel || 'RBF'} Kernel`;
	if (chipMargin) chipMargin.textContent = `Margin: ${marginSign}${activeMargin.toFixed(2)}`;
	if (chipVectors) chipVectors.textContent = `SV: ${(ksvm.total_support_vectors || 0).toLocaleString()}`;

	// Card summary
	const activeTag = document.getElementById('ksvmActiveTag');
	if (activeTag) activeTag.textContent = ksvm.active_model || 'Scaled Pipeline';

	const kernelVal = document.getElementById('ksvmKernelVal');
	if (kernelVal) kernelVal.textContent = ksvm.kernel || 'RBF';

	const paramsVal = document.getElementById('ksvmParamsVal');
	if (paramsVal) paramsVal.textContent = `C=${ksvm.c_param || 10} · γ=${ksvm.gamma || 'scale'}`;

	const vectorsVal = document.getElementById('ksvmVectorsVal');
	if (vectorsVal) vectorsVal.textContent = (ksvm.total_support_vectors || 0).toLocaleString();

	const maxMarginVal = document.getElementById('ksvmMaxMarginVal');
	if (maxMarginVal) maxMarginVal.textContent = `${marginSign}${activeMargin.toFixed(2)}`;

	// Decision function margin bars
	const classes = ['glioma', 'meningioma', 'pituitary', 'notumor'];
	classes.forEach(c => {
		const score = decisionScores[c];
		const scoreEl = document.getElementById(`df-${c}`);
		const barEl = document.getElementById(`dfBar-${c}`);
		if (scoreEl && score !== undefined) {
			const s = score >= 0 ? `+${score.toFixed(2)}` : score.toFixed(2);
			scoreEl.textContent = s;
			scoreEl.style.color = score >= 0 ? '#10b981' : '#6b7280';
		}
		if (barEl && score !== undefined) {
			// Normalize [-4, +4] to [5%, 95%]
			const pct = Math.max(5, Math.min(95, ((score + 4) / 8) * 100));
			barEl.style.width = `${pct}%`;
			if (score >= 0) {
				barEl.style.background = 'linear-gradient(90deg, #3568e8, #10b981)';
			} else {
				barEl.style.background = 'linear-gradient(90deg, #9ca3af, #6b7280)';
			}
		}
	});

	// Support vectors breakdown
	const svByClass = ksvm.support_vectors_by_class || {};
	classes.forEach(c => {
		const svEl = document.getElementById(`sv-${c}`);
		if (svEl && svByClass[c] !== undefined) {
			svEl.textContent = svByClass[c].toLocaleString();
		}
	});

	// 32 Selected FastICA Features
	const sparkContainer = document.getElementById('featureSparkBar');
	if (sparkContainer && Array.isArray(ksvm.selected_features)) {
		sparkContainer.innerHTML = '';
		const features = ksvm.selected_features;
		const maxAbs = Math.max(...features.map(f => Math.abs(f)), 0.001);
		features.forEach((val, idx) => {
			const bar = document.createElement('div');
			bar.className = 'spark-bar';
			const heightPercent = Math.max(12, Math.round((Math.abs(val) / maxAbs) * 100));
			bar.style.height = `${heightPercent}%`;
			if (val < 0) {
				bar.style.background = '#8668df';
			}
			bar.title = `ICA Component #${idx + 1}: ${val}`;
			sparkContainer.appendChild(bar);
		});
	}
}

analyzeBtn.addEventListener('click', async () => {
	if (!selectedFile) return;
	analyzeBtn.disabled = true;
	analyzeBtn.innerHTML = 'Analyzing <span>…</span>';
	scanState.textContent = 'Processing';
	try {
		const form = new FormData();
		form.append('file', selectedFile);
		const chosenEngine = modelSelect ? modelSelect.value : 'ksvm';
		form.append('model', chosenEngine);
		const response = await fetch(`${API_BASE_URL}/analyze`, {
			method: 'POST',
			body: form,
		});
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
		updateKsvmDiagnostics(result.ksvm, result.label);
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
		script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/4.2.1/jspdf.umd.min.js';
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
		const ksvm = result.ksvm || {};

		report.setFontSize(20);
		report.text('NeuroLens MRI Analysis Report', 14, 20);
		report.setFontSize(11);
		report.text(`Scan: ${selectedFile?.name || 'MRI image'}`, 14, 34);
		report.text(`Predicted class: ${label}`, 14, 43);
		report.text(`Confidence: ${(result.confidence * 100).toFixed(1)}%`, 14, 52);
		report.text(`Status: ${status}`, 14, 61);
		report.text(`Classifier: ${ksvm.active_model || 'Kernel SVM (RBF)'}`, 14, 70);
		report.text(`Kernel function: ${ksvm.kernel || 'RBF'} (C=${ksvm.c_param || 10}, γ=${ksvm.gamma || 'scale'})`, 14, 78);
		if (ksvm.total_support_vectors) {
			report.text(`Active Support Vectors: ${ksvm.total_support_vectors.toLocaleString()}`, 14, 86);
		}

		report.setFontSize(13);
		report.text('Class Probabilities & Decision Margins', 14, 100);
		report.setFontSize(10);
		const classNames = { glioma: 'Glioma', meningioma: 'Meningioma', notumor: 'No tumor', pituitary: 'Pituitary' };
		const dfScores = ksvm.decision_scores || {};
		Object.entries(result.probabilities || {}).forEach(([key, value], index) => {
			const df = dfScores[key] !== undefined ? ` (Margin: ${dfScores[key] >= 0 ? '+' : ''}${dfScores[key].toFixed(2)})` : '';
			report.text(`${classNames[key] || key}: ${(value * 100).toFixed(1)}%${df}`, 14, 110 + index * 7);
		});

		report.setFontSize(9);
		report.text(
			'Visual outputs are review aids, not a substitute for clinical interpretation. The highlight and segmentation are heuristic; the feature map shows the ICA reconstruction.',
			14,
			148,
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
		report.save('neurolens-ksvm-mri-report.pdf');
	} catch (error) {
		alert(error.message || 'Could not create the PDF report.');
	} finally {
		button.disabled = false;
		button.textContent = '↓ Download PDF report';
	}
});
