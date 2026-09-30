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

/*
 * Render backend URL
 * Frontend: Vercel
 * Backend: Render
 */
const API_BASE_URL = 'https://brain-tumor-mri-46od.onrender.com';

/* =========================================================
   ICA FEATURE MAP CARD
   ========================================================= */

const featureCard = document.createElement('article');

featureCard.className = 'output-card';

featureCard.innerHTML = `
	<div class="output-card-head">
		<b>4. ICA feature map</b>
		<span>Feature reconstruction</span>
	</div>

	<div class="output-image" id="featureMapOutput">
		<div class="output-placeholder">
			Available after analysis
		</div>
	</div>

	<p>
		Image reconstructed from the independent components used by the classifier.
	</p>
`;

const outputGrid = document.querySelector('.output-grid');

if (outputGrid) {
	outputGrid.appendChild(featureCard);
}

/* =========================================================
   FILE UPLOAD
   ========================================================= */

fileInput.addEventListener('change', event => {
	selectedFile = event.target.files[0];

	if (!selectedFile) {
		return;
	}

	/* Maximum file size: 10 MB */
	if (selectedFile.size > 10 * 1024 * 1024) {
		fileName.textContent = 'File exceeds 10 MB';
		analyzeBtn.disabled = true;
		selectedFile = null;
		return;
	}

	fileName.textContent = selectedFile.name;
	scanState.textContent = 'Ready';

	analyzeBtn.disabled = false;

	const reader = new FileReader();

	reader.onload = loadEvent => {
		preview.style.backgroundImage =
			`url(${loadEvent.target.result})`;

		preview.classList.add('has-image');

		setOutput(
			'originalOutput',
			loadEvent.target.result
		);
	};

	reader.readAsDataURL(selectedFile);
});

/* =========================================================
   SET OUTPUT IMAGE
   ========================================================= */

function setOutput(id, source) {
	const element = document.getElementById(id);

	if (!element) {
		return;
	}

	element.innerHTML = '';

	if (!source) {
		const placeholder = document.createElement('div');

		placeholder.className = 'output-placeholder';
		placeholder.textContent = 'Image unavailable';

		element.appendChild(placeholder);

		return;
	}

	const image = document.createElement('img');

	image.src = source;
	image.alt = `${id.replace('Output', '')} MRI output`;

	element.appendChild(image);
}

/* =========================================================
   UPDATE CLASS PROBABILITIES
   ========================================================= */

function updateProbabilities(probabilities) {
	document.querySelectorAll('.prob-row').forEach(row => {
		const labelElement = row.querySelector('span');

		if (!labelElement) {
			return;
		}

		const label = labelElement.textContent.trim();

		const key =
			label === 'No tumor'
				? 'notumor'
				: label.toLowerCase();

		const value =
			(probabilities?.[key] || 0) * 100;

		const progress = row.querySelector('i b');
		const percentage = row.querySelector('strong');

		if (progress) {
			progress.style.width = `${value}%`;
		}

		if (percentage) {
			percentage.textContent =
				`${value.toFixed(1)}%`;
		}
	});
}

/* =========================================================
   MRI ANALYSIS
   ========================================================= */

analyzeBtn.addEventListener('click', async () => {
	if (!selectedFile) {
		return;
	}

	analyzeBtn.disabled = true;
	analyzeBtn.innerHTML =
		'Analyzing <span>…</span>';

	scanState.textContent = 'Processing';

	try {
		/*
		 * Create multipart form data
		 */
		const form = new FormData();

		form.append('file', selectedFile);

		/*
		 * Send MRI image to Render Python API
		 */
		const response = await fetch(
			`${API_BASE_URL}/analyze`,
			{
				method: 'POST',
				body: form
			}
		);

		if (!response.ok) {
			throw new Error(
				`API unavailable (${response.status})`
			);
		}

		/*
		 * Read JSON response
		 */
		latestResult = await response.json();

		const result = latestResult;

		/* =====================================================
		   SHOW RESULT SECTION
		   ===================================================== */

		resultEmpty.hidden = true;
		resultContent.hidden = false;

		/* =====================================================
		   TUMOR STATUS
		   ===================================================== */

		const isTumorPositive =
			Boolean(result.tumor_detected);

		resultBadge.textContent =
			isTumorPositive
				? 'Tumor positive'
				: 'Tumor negative';

		resultBadge.style.background =
			isTumorPositive
				? '#fef2f2'
				: '#ecfdf5';

		resultBadge.style.color =
			isTumorPositive
				? '#b91c1c'
				: '#047857';

		/* =====================================================
		   PREDICTED CLASS
		   ===================================================== */

		const resultLabel =
			document.getElementById('resultLabel');

		if (resultLabel) {
			resultLabel.textContent =
				result.label === 'notumor'
					? 'No tumor'
					: result.label
						? result.label[0].toUpperCase() +
						  result.label.slice(1)
						: 'Unknown';
		}

		/* =====================================================
		   RESULT STATUS
		   ===================================================== */

		const resultStatus =
			document.getElementById('resultStatus');

		if (resultStatus) {
			resultStatus.textContent =
				isTumorPositive
					? 'Tumor positive · review visual aids'
					: 'Tumor negative';
		}

		/* =====================================================
		   TUMOR STATUS INDICATOR
		   ===================================================== */

		if (tumorStatus) {
			tumorStatus.innerHTML =
				isTumorPositive
					? `
						<span
							style="
								display:inline-block;
								width:8px;
								height:8px;
								border-radius:50%;
								background:#dc2626;
							">
						</span>
						Tumor status: POSITIVE
					`
					: `
						<span
							style="
								display:inline-block;
								width:8px;
								height:8px;
								border-radius:50%;
								background:#22c55e;
							">
						</span>
						Tumor status: NEGATIVE
					`;
		}

		/* =====================================================
		   CONFIDENCE
		   ===================================================== */

		const confidence =
			Number(result.confidence || 0);

		const confidencePercentage =
			confidence * 100;

		const confidenceValue =
			document.getElementById('confidenceValue');

		if (confidenceValue) {
			confidenceValue.textContent =
				`${confidencePercentage.toFixed(1)}%`;
		}

		const confidenceBar =
			document.getElementById('confidenceBar');

		if (confidenceBar) {
			confidenceBar.style.width =
				`${confidencePercentage}%`;
		}

		/* =====================================================
		   CLASS PROBABILITIES
		   ===================================================== */

		updateProbabilities(
			result.probabilities
		);

		/* =====================================================
		   MRI VISUAL OUTPUTS
		   ===================================================== */

		if (result.images) {
			setOutput(
				'originalOutput',
				result.images.original
			);

			setOutput(
				'highlightOutput',
				result.images.highlight
			);

			setOutput(
				'segmentationOutput',
				result.images.segmentation
			);

			setOutput(
				'featureMapOutput',
				result.images.feature_map
			);
		}

		scanState.textContent = 'Complete';

	} catch (error) {

		console.error(
			'NeuroLens API error:',
			error
		);

		resultEmpty.hidden = false;
		resultContent.hidden = true;

		const emptyTitle =
			resultEmpty.querySelector('b');

		const emptyDescription =
			resultEmpty.querySelector('p');

		if (emptyTitle) {
			emptyTitle.textContent =
				'Analysis service unavailable';
		}

		if (emptyDescription) {
			emptyDescription.textContent =
				'The MRI analysis service is currently unavailable. Please try again in a moment.';
		}

		if (tumorStatus) {
			tumorStatus.innerHTML = `
				<span
					style="
						display:inline-block;
						width:8px;
						height:8px;
						border-radius:50%;
						background:#d1d5db;
					">
				</span>
				Tumor status: —
			`;
		}

		scanState.textContent =
			'API offline';

	} finally {

		analyzeBtn.disabled = false;

		analyzeBtn.innerHTML =
			'Run analysis <span>→</span>';
	}
});

/* =========================================================
   DARK / LIGHT THEME
   ========================================================= */

themeToggle.addEventListener('click', () => {
	const html =
		document.documentElement;

	const dark =
		html.dataset.theme === 'dark';

	html.dataset.theme =
		dark ? 'light' : 'dark';

	themeToggle.textContent =
		dark ? '◐' : '☼';
});

/* =========================================================
   PDF LIBRARY
   ========================================================= */

function loadPdfLibrary() {

	/*
	 * If jsPDF is already available,
	 * don't load it again.
	 */
	if (window.jspdf?.jsPDF) {
		return Promise.resolve();
	}

	return new Promise((resolve, reject) => {

		const script =
			document.createElement('script');

		/*
		 * Load jsPDF from CDN.
		 * This works with the static Vercel frontend.
		 */
		script.src =
			'https://cdnjs.cloudflare.com/ajax/libs/jspdf/4.2.1/jspdf.umd.min.js';

		script.onload = resolve;

		script.onerror = () =>
			reject(
				new Error(
					'Could not load the PDF generator.'
				)
			);

		document.head.appendChild(script);
	});
}

/* =========================================================
   DOWNLOAD PDF REPORT
   ========================================================= */

document
	.getElementById('downloadBtn')
	.addEventListener(
		'click',
		async event => {

			if (!latestResult) {
				return;
			}

			const button =
				event.currentTarget;

			button.disabled = true;

			button.textContent =
				'Preparing report...';

			try {

				/* Load jsPDF */
				await loadPdfLibrary();

				const { jsPDF } =
					window.jspdf;

				const report =
					new jsPDF();

				const result =
					latestResult;

				const label =
					result.label === 'notumor'
						? 'No tumor'
						: result.label || 'Unknown';

				const status =
					result.tumor_detected
						? 'Tumor status: POSITIVE'
						: 'Tumor status: NEGATIVE';

				/* =================================================
				   REPORT TITLE
				   ================================================= */

				report.setFontSize(20);

				report.text(
					'NeuroLens MRI Analysis Report',
					14,
					20
				);

				/* =================================================
				   BASIC INFORMATION
				   ================================================= */

				report.setFontSize(11);

				report.text(
					`Scan: ${selectedFile?.name || 'MRI image'}`,
					14,
					34
				);

				report.text(
					`Predicted class: ${label}`,
					14,
					43
				);

				report.text(
					`Confidence: ${(Number(result.confidence || 0) * 100).toFixed(1)}%`,
					14,
					52
				);

				report.text(
					`Status: ${status}`,
					14,
					61
				);

				report.text(
					'Method: ICA feature extraction + Kernel SVM',
					14,
					70
				);

				/* =================================================
				   CLASS PROBABILITIES
				   ================================================= */

				report.setFontSize(13);

				report.text(
					'Class probabilities',
					14,
					84
				);

				report.setFontSize(10);

				const classNames = {
					glioma: 'Glioma',
					meningioma: 'Meningioma',
					notumor: 'No tumor',
					pituitary: 'Pituitary'
				};

				Object.entries(
					result.probabilities || {}
				).forEach(
					([key, value], index) => {

						report.text(
							`${classNames[key] || key}: ${(Number(value) * 100).toFixed(1)}%`,
							14,
							94 + index * 7
						);
					}
				);

				/* =================================================
				   DISCLAIMER
				   ================================================= */

				report.setFontSize(9);

				report.text(
					'Visual outputs are review aids, not a substitute for clinical interpretation. The highlight and segmentation are heuristic; the feature map shows the ICA reconstruction.',
					14,
					132,
					{
						maxWidth: 182
					}
				);

				/* =================================================
				   SECOND PAGE
				   ================================================= */

				report.addPage();

				report.setFontSize(16);

				report.text(
					'Model visual outputs',
					14,
					20
				);

				/* =================================================
				   VISUAL OUTPUTS
				   ================================================= */

				const visuals = [
					[
						'Original MRI',
						result.images?.original
					],
					[
						'Tumor highlighting',
						result.images?.highlight
					],
					[
						'Segmentation mask',
						result.images?.segmentation
					],
					[
						'ICA feature map',
						result.images?.feature_map
					]
				];

				visuals.forEach(
					([title, image], index) => {

						const x =
							14 +
							(index % 2) * 94;

						const y =
							31 +
							Math.floor(index / 2) * 124;

						report.setFontSize(10);

						report.text(
							title,
							x,
							y
						);

						if (image) {
							try {

								report.addImage(
									image,
									'PNG',
									x,
									y + 4,
									82,
									82,
									undefined,
									'FAST'
								);

							} catch (imageError) {

								console.warn(
									'Could not add image to PDF:',
									imageError
								);
							}
						}
					}
				);

				/* =================================================
				   SAVE REPORT
				   ================================================= */

				report.save(
					'neurolens-mri-report.pdf'
				);

			} catch (error) {

				console.error(
					'PDF generation error:',
					error
				);

				alert(
					error.message ||
					'Could not create the PDF report.'
				);

			} finally {

				button.disabled = false;

				button.textContent =
					'↓ Download PDF report';
			}
		}
	);
