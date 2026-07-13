// Application controller for Bank Marketing Decision Tree Predictor & Explorer

document.addEventListener('DOMContentLoaded', () => {
    let modelData = null;
    let activePathNodeIds = new Set();
    
    // Zoom and pan state
    let zoomScale = 1.0;
    let panX = 0;
    let panY = 0;
    let isDragging = false;
    let startDragX = 0;
    let startDragY = 0;

    // DOM Elements
    const predictorForm = document.getElementById('predictor-form');
    const resultsCard = document.getElementById('results-card');
    const economicAccordionBtn = document.getElementById('economic-accordion-btn');
    const economicAccordionContent = document.getElementById('economic-accordion-content');
    
    const predictionPercentage = document.getElementById('prediction-percentage');
    const predictionGaugeFill = document.getElementById('prediction-gauge-fill');
    const predictionOutcome = document.getElementById('prediction-outcome');
    const predictionExplanation = document.getElementById('prediction-explanation');
    const pathStepsContainer = document.getElementById('path-steps-container');
    
    const treeSvg = document.getElementById('tree-svg');
    const treeViewport = document.getElementById('tree-viewport-container');

    // Zoom Buttons
    const zoomInBtn = document.getElementById('zoom-in-btn');
    const zoomOutBtn = document.getElementById('zoom-out-btn');
    const zoomResetBtn = document.getElementById('zoom-reset-btn');

    // Initialize Accordion
    economicAccordionBtn.addEventListener('click', () => {
        economicAccordionBtn.classList.toggle('active');
        economicAccordionContent.classList.toggle('open');
    });

    // Load Model JSON Data
    fetch('model_data.json')
        .then(response => response.json())
        .then(data => {
            modelData = data;
            initializeDashboard(data);
        })
        .catch(err => {
            console.error('Error loading model assets:', err);
            alert('Failed to load model data. Make sure you run train_model.py first.');
        });

    function initializeDashboard(data) {
        // 1. Populate Dropdown Select Fields
        populateDropdowns(data.feature_categories);

        // 2. Set Model Metrics Cards
        document.getElementById('metric-accuracy').textContent = (data.metrics.accuracy * 100).toFixed(1) + '%';
        document.getElementById('metric-f1').textContent = data.metrics.f1_score.toFixed(2);
        document.getElementById('metric-precision').textContent = (data.metrics.precision * 100).toFixed(1) + '%';
        document.getElementById('metric-recall').textContent = (data.metrics.recall * 100).toFixed(1) + '%';

        // 3. Set Confusion Matrix
        // CM structure: [[TN, FP], [FN, TP]]
        const cm = data.confusion_matrix;
        document.getElementById('cell-tn').textContent = cm[0][0].toLocaleString();
        document.getElementById('cell-fp').textContent = cm[0][1].toLocaleString();
        document.getElementById('cell-fn').textContent = cm[1][0].toLocaleString();
        document.getElementById('cell-tp').textContent = cm[1][1].toLocaleString();

        // 4. Render Feature Importance Bars
        renderFeatureImportance(data.feature_importances);

        // 5. Draw the initial Decision Tree
        drawDecisionTree(data.decision_tree);
    }

    function populateDropdowns(categories) {
        for (const [featureName, options] of Object.entries(categories)) {
            const selectId = `select-${featureName.replace('.', '-')}`;
            const selectEl = document.getElementById(selectId);
            if (selectEl) {
                selectEl.innerHTML = '';
                options.forEach(opt => {
                    const optionEl = document.createElement('option');
                    optionEl.value = opt;
                    optionEl.textContent = opt;
                    
                    // Select smart defaults
                    if (featureName === 'marital' && opt === 'single') optionEl.selected = true;
                    if (featureName === 'education' && opt === 'university.degree') optionEl.selected = true;
                    if (featureName === 'default' && opt === 'no') optionEl.selected = true;
                    if (featureName === 'housing' && opt === 'yes') optionEl.selected = true;
                    if (featureName === 'loan' && opt === 'no') optionEl.selected = true;
                    if (featureName === 'poutcome' && opt === 'nonexistent') optionEl.selected = true;
                    if (featureName === 'contact' && opt === 'cellular') optionEl.selected = true;
                    if (featureName === 'month' && opt === 'may') optionEl.selected = true;
                    
                    selectEl.appendChild(optionEl);
                });
            }
        }
    }

    function renderFeatureImportance(importances) {
        const listEl = document.getElementById('feature-importance-list');
        listEl.innerHTML = '';
        
        // Show top 6 features for clean UI
        const topImportances = importances.slice(0, 6);
        const maxVal = topImportances[0].importance;

        topImportances.forEach(item => {
            const pct = (item.importance / maxVal) * 100;
            const row = document.createElement('div');
            row.className = 'feature-bar-item';
            row.innerHTML = `
                <span class="feature-name" title="${item.feature}">${formatFeatureName(item.feature)}</span>
                <div class="feature-bar-track">
                    <div class="feature-bar-fill" style="width: 0%"></div>
                </div>
                <span class="feature-val">${(item.importance * 100).toFixed(1)}%</span>
            `;
            listEl.appendChild(row);
            
            // Trigger animation after append
            setTimeout(() => {
                row.querySelector('.feature-bar-fill').style.width = `${pct}%`;
            }, 100);
        });
    }

    function formatFeatureName(name) {
        // Prettify encoded feature names
        return name
            .replace('_', ': ')
            .replace('emp.var.rate', 'Employment Var. Rate')
            .replace('cons.price.idx', 'Consumer Price Index')
            .replace('cons.conf.idx', 'Consumer Confidence')
            .replace('euribor3m', 'Euribor 3M')
            .replace('nr.employed', 'No. Employees')
            .replace('pdays', 'Days Since Contact')
            .replace('previous', 'Prev Contacts')
            .replace('campaign', 'Campaign Contacts');
    }

    // Handle Form Submit (Prediction)
    predictorForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!modelData) return;

        const formData = new FormData(predictorForm);
        const inputs = {};
        for (const [key, value] of formData.entries()) {
            if (key === 'age' || key === 'campaign' || key === 'pdays' || key === 'previous') {
                inputs[key] = parseInt(value, 10);
            } else if (key === 'emp.var.rate' || key === 'cons.price.idx' || key === 'cons.conf.idx' || key === 'euribor3m' || key === 'nr.employed') {
                inputs[key] = parseFloat(value);
            } else {
                inputs[key] = value;
            }
        }

        // Run Local Decision Path Prediction
        const prediction = predictDecisionTree(modelData.decision_tree, inputs);

        // Display Results Card
        displayPrediction(prediction);
    });

    function getFeatureValue(featureName, inputs) {
        // If it's a direct numerical feature
        if (inputs.hasOwnProperty(featureName)) {
            return inputs[featureName];
        }
        
        // If it is a one-hot encoded categorical feature (e.g. job_admin.)
        for (const [catName, categories] of Object.entries(modelData.feature_categories)) {
            if (featureName.startsWith(catName + '_')) {
                const optVal = featureName.substring(catName.length + 1);
                // Compare input value with encoded option
                return inputs[catName] === optVal ? 1 : 0;
            }
        }
        return 0;
    }

    function predictDecisionTree(node, inputs, path = []) {
        path.push(node);
        
        if (node.is_leaf) {
            // Calculate probability of yes
            const total = node.value[0] + node.value[1];
            const probYes = total > 0 ? (node.value[1] / total) : 0;
            return {
                outcome: node.class_name,
                probability: probYes,
                path: path
            };
        }

        const featureVal = getFeatureValue(node.feature_name, inputs);
        
        // Record condition text
        let conditionText = '';
        if (node.feature_name.includes('_')) {
            const parts = node.feature_name.split('_');
            const field = parts[0];
            const category = parts.slice(1).join('_');
            const isActive = featureVal === 1;
            conditionText = `${field} is "${category}"? (${isActive ? 'Yes' : 'No'})`;
        } else {
            conditionText = `${formatFeatureName(node.feature_name)} (${featureVal}) ${featureVal <= node.threshold ? '≤' : '>'} ${node.threshold}`;
        }
        
        node.decision_explanation = conditionText;

        if (featureVal <= node.threshold) {
            node.taken_branch = 'left';
            return predictDecisionTree(node.left, inputs, path);
        } else {
            node.taken_branch = 'right';
            return predictDecisionTree(node.right, inputs, path);
        }
    }

    function displayPrediction(pred) {
        resultsCard.classList.remove('hidden');
        
        const probPct = Math.round(pred.probability * 100);
        predictionPercentage.textContent = `${probPct}%`;
        
        // Rotate vintage speedometer needle
        // Scale is a semicircle from left to right (from -90deg to +90deg)
        const needle = document.getElementById('speedometer-needle');
        if (needle) {
            const angle = -90 + (pred.probability * 180);
            needle.style.transform = `rotate(${angle}deg)`;
        }
        
        // Remove old outcome classes
        predictionOutcome.className = '';
        
        if (pred.outcome === 'yes') {
            predictionOutcome.textContent = 'Term Deposit Subscribed';
            predictionOutcome.classList.add('yes');
            predictionExplanation.textContent = `High match profile! The model estimates a ${probPct}% probability that this client will subscribe to the term deposit.`;
        } else {
            predictionOutcome.textContent = 'Will Decline Subscription';
            predictionOutcome.classList.add('no');
            predictionExplanation.textContent = `Lower match profile. The model estimates only a ${probPct}% subscription probability based on current demographics & indicators.`;
        }

        // Save active path node IDs for SVG highlighting
        activePathNodeIds = new Set(pred.path.map(n => n.node_id));

        // Populate steps in path tracker
        pathStepsContainer.innerHTML = '';
        pred.path.forEach((node, idx) => {
            if (node.is_leaf) {
                const step = document.createElement('div');
                step.className = 'path-step highlight';
                step.style.borderLeftColor = pred.outcome === 'yes' ? 'var(--status-yes)' : 'var(--status-no)';
                step.innerHTML = `➔ <strong>Leaf Node</strong>: Predicted "${node.class_name.toUpperCase()}" (n=${node.samples})`;
                pathStepsContainer.appendChild(step);
            } else {
                const step = document.createElement('div');
                step.className = 'path-step highlight';
                step.innerHTML = `Step ${idx+1}: Split on <strong>${formatFeatureName(node.feature_name)}</strong> <br> &nbsp;&nbsp;&nbsp;&nbsp;(${node.decision_explanation})`;
                pathStepsContainer.appendChild(step);
            }
        });

        // Update Tree SVG Highlights
        updateTreeHighlights();
    }

    // Helper functions for Layout Calculation
    function getLeafCount(node) {
        if (node.is_leaf) return 1;
        return getLeafCount(node.left) + getLeafCount(node.right);
    }

    function getMaxDepth(node) {
        if (node.is_leaf) return 1;
        return 1 + Math.max(getMaxDepth(node.left), getMaxDepth(node.right));
    }

    // SVG Tidy tree coordinates mapping
    function assignCoordinates(node, depth, startX, leafWidth) {
        node.depth = depth;
        node.y = depth * 110 + 60;
        
        if (node.is_leaf) {
            node.x = startX + leafWidth / 2;
            return leafWidth;
        } else {
            const leftLeaves = getLeafCount(node.left);
            const rightLeaves = getLeafCount(node.right);
            
            const leftWidth = leftLeaves * leafWidth;
            const rightWidth = rightLeaves * leafWidth;
            
            assignCoordinates(node.left, depth + 1, startX, leafWidth);
            assignCoordinates(node.right, depth + 1, startX + leftWidth, leafWidth);
            
            node.x = (node.left.x + node.right.x) / 2;
            return leftWidth + rightWidth;
        }
    }

    // Draw Decision Tree SVG
    function drawDecisionTree(rootNode) {
        const leafCount = getLeafCount(rootNode);
        const maxDepth = getMaxDepth(rootNode);
        
        // Sizing variables
        const leafWidth = 60; // Spacing between leaf nodes
        const totalWidth = leafCount * leafWidth;
        const totalHeight = maxDepth * 110 + 100;
        
        // Set up SVG container attributes
        treeSvg.setAttribute('viewBox', `0 0 ${totalWidth} ${totalHeight}`);
        
        // Clear SVG
        treeSvg.innerHTML = '';
        
        // Define filters and gradients in SVG defs
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        defs.innerHTML = `
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="6" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
        `;
        treeSvg.appendChild(defs);

        // Assign Coordinates to all nodes recursively
        assignCoordinates(rootNode, 0, 0, leafWidth);

        // Group for rendering elements so we can pan and zoom
        const mainGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        mainGroup.setAttribute('id', 'tree-transform-group');
        treeSvg.appendChild(mainGroup);

        // Draw Links
        renderLinks(rootNode, mainGroup);
        // Draw Nodes
        renderNodes(rootNode, mainGroup);
        
        // Apply default center pan
        resetZoomPan(totalWidth, totalHeight);

        // Setup Zoom & Drag Listeners
        setupViewportInteractions(totalWidth, totalHeight);
    }

    function renderLinks(node, group) {
        if (node.is_leaf) return;

        // Draw link to left child
        const linkLeft = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const dLeft = `M ${node.x} ${node.y} C ${node.x} ${(node.y + node.left.y)/2}, ${node.left.x} ${(node.y + node.left.y)/2}, ${node.left.x} ${node.left.y}`;
        linkLeft.setAttribute('d', dLeft);
        linkLeft.setAttribute('class', 'node-link');
        linkLeft.setAttribute('id', `link-left-${node.node_id}`);
        group.appendChild(linkLeft);

        // Label for left split (<= threshold)
        const labelLeft = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        labelLeft.setAttribute('x', (node.x + node.left.x) / 2 - 10);
        labelLeft.setAttribute('y', (node.y + node.left.y) / 2 - 4);
        labelLeft.setAttribute('fill', 'var(--text-muted)');
        labelLeft.setAttribute('font-size', '9px');
        labelLeft.setAttribute('text-anchor', 'end');
        labelLeft.textContent = 'Yes';
        group.appendChild(labelLeft);

        // Draw link to right child
        const linkRight = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const dRight = `M ${node.x} ${node.y} C ${node.x} ${(node.y + node.right.y)/2}, ${node.right.x} ${(node.y + node.right.y)/2}, ${node.right.x} ${node.right.y}`;
        linkRight.setAttribute('d', dRight);
        linkRight.setAttribute('class', 'node-link');
        linkRight.setAttribute('id', `link-right-${node.node_id}`);
        group.appendChild(linkRight);

        // Label for right split (> threshold)
        const labelRight = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        labelRight.setAttribute('x', (node.x + node.right.x) / 2 + 10);
        labelRight.setAttribute('y', (node.y + node.right.y) / 2 - 4);
        labelRight.setAttribute('fill', 'var(--text-muted)');
        labelRight.setAttribute('font-size', '9px');
        labelRight.setAttribute('text-anchor', 'start');
        labelRight.textContent = 'No';
        group.appendChild(labelRight);

        // Recurse
        renderLinks(node.left, group);
        renderLinks(node.right, group);
    }

    function renderNodes(node, group) {
        const nodeG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        
        // Node circle representation
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', node.x);
        circle.setAttribute('cy', node.y);
        
        let circleClass = 'node-circle';
        if (node.is_leaf) {
            circleClass += ` leaf leaf-${node.class_name}`;
            circle.setAttribute('r', 16);
        } else {
            circle.setAttribute('r', 14);
        }
        
        circle.setAttribute('class', circleClass);
        circle.setAttribute('id', `node-circle-${node.node_id}`);
        
        // Tooltip description
        const tooltip = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        const total = node.value[0] + node.value[1];
        const yesPct = ((node.value[1] / total) * 100).toFixed(1);
        if (node.is_leaf) {
            tooltip.textContent = `Leaf Node #${node.node_id}\nSamples: ${node.samples}\nClass: ${node.class_name.toUpperCase()}\nYes Ratio: ${yesPct}%`;
        } else {
            tooltip.textContent = `Split Node #${node.node_id}\nSplit Feature: ${node.feature_name}\nThreshold: ${node.threshold}\nSamples: ${node.samples}\nYes Ratio: ${yesPct}%`;
        }
        circle.appendChild(tooltip);
        nodeG.appendChild(circle);

        // Labels
        if (node.is_leaf) {
            const labelText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            labelText.setAttribute('x', node.x);
            labelText.setAttribute('y', node.y + 4);
            labelText.setAttribute('text-anchor', 'middle');
            labelText.setAttribute('class', 'node-value-text');
            labelText.textContent = node.class_name.toUpperCase();
            nodeG.appendChild(labelText);
        } else {
            // Split variable label above node
            const labelText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            labelText.setAttribute('x', node.x);
            labelText.setAttribute('y', node.y - 20);
            labelText.setAttribute('text-anchor', 'middle');
            labelText.setAttribute('class', 'node-label');
            
            // Clean display feature name
            let dispName = formatFeatureName(node.feature_name);
            if (dispName.length > 15) dispName = dispName.substring(0, 13) + '...';
            
            labelText.textContent = dispName;
            nodeG.appendChild(labelText);
            
            // Threshold split criterion inside/below node
            const subtext = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            subtext.setAttribute('x', node.x);
            subtext.setAttribute('y', node.y + 4);
            subtext.setAttribute('text-anchor', 'middle');
            subtext.setAttribute('class', 'node-subtext');
            subtext.setAttribute('fill', '#fff');
            subtext.textContent = `≤ ${node.threshold}`;
            nodeG.appendChild(subtext);
        }

        group.appendChild(nodeG);

        if (!node.is_leaf) {
            renderNodes(node.left, group);
            renderNodes(node.right, group);
        }
    }

    function updateTreeHighlights() {
        // Reset all classes
        document.querySelectorAll('.node-circle').forEach(c => c.classList.remove('active-path'));
        document.querySelectorAll('.node-link').forEach(l => l.classList.remove('active-path'));

        if (activePathNodeIds.size === 0) return;

        // Traverse modelData tree to find which links are active
        function highlightPaths(node) {
            if (node.is_leaf) {
                const el = document.getElementById(`node-circle-${node.node_id}`);
                if (el) el.classList.add('active-path');
                return;
            }

            const el = document.getElementById(`node-circle-${node.node_id}`);
            if (el) el.classList.add('active-path');

            // Determine if left or right was taken
            if (activePathNodeIds.has(node.left.node_id)) {
                const link = document.getElementById(`link-left-${node.node_id}`);
                if (link) link.classList.add('active-path');
                highlightPaths(node.left);
            } else if (activePathNodeIds.has(node.right.node_id)) {
                const link = document.getElementById(`link-right-${node.node_id}`);
                if (link) link.classList.add('active-path');
                highlightPaths(node.right);
            }
        }

        if (modelData) {
            highlightPaths(modelData.decision_tree);
        }
    }

    // Viewport zoom & pan handlers
    function setupViewportInteractions(totalWidth, totalHeight) {
        const transGroup = document.getElementById('tree-transform-group');

        function updateTransform() {
            transGroup.setAttribute('transform', `translate(${panX}, ${panY}) scale(${zoomScale})`);
        }

        // Zoom button bindings
        zoomInBtn.addEventListener('click', () => {
            zoomScale *= 1.2;
            updateTransform();
        });

        zoomOutBtn.addEventListener('click', () => {
            zoomScale /= 1.2;
            updateTransform();
        });

        zoomResetBtn.addEventListener('click', () => {
            resetZoomPan(totalWidth, totalHeight);
        });

        // Mouse pan events
        treeViewport.addEventListener('mousedown', (e) => {
            // Ignore if clicked on a button or interactive child
            if (e.target.tagName === 'BUTTON') return;
            isDragging = true;
            startDragX = e.clientX - panX;
            startDragY = e.clientY - panY;
            treeSvg.style.cursor = 'grabbing';
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            panX = e.clientX - startDragX;
            panY = e.clientY - startDragY;
            updateTransform();
        });

        window.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                treeSvg.style.cursor = 'grab';
            }
        });

        // Mouse wheel zoom
        treeViewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomFactor = 1.1;
            if (e.deltaY < 0) {
                zoomScale *= zoomFactor;
            } else {
                zoomScale /= zoomFactor;
            }
            updateTransform();
        });
    }

    function resetZoomPan(totalWidth, totalHeight) {
        const transGroup = document.getElementById('tree-transform-group');
        const containerWidth = treeViewport.clientWidth;
        
        // Calculate initial zoom to fit horizontal space with some padding
        zoomScale = Math.min(1.0, (containerWidth - 40) / totalWidth);
        if (zoomScale < 0.2) zoomScale = 0.2; // Min baseline
        
        // Center horizontally
        panX = (containerWidth - totalWidth * zoomScale) / 2;
        panY = 30; // 30px padding top
        
        transGroup.setAttribute('transform', `translate(${panX}, ${panY}) scale(${zoomScale})`);
    }

    // Window resize handler to update centered pan
    window.addEventListener('resize', () => {
        if (modelData) {
            const leafCount = getLeafCount(modelData.decision_tree);
            const leafWidth = 60;
            const totalWidth = leafCount * leafWidth;
            const maxDepth = getMaxDepth(modelData.decision_tree);
            const totalHeight = maxDepth * 110 + 100;
            resetZoomPan(totalWidth, totalHeight);
        }
    });
});
