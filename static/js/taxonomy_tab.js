document.addEventListener("DOMContentLoaded", function () {
    // -- State ---------------------------------------------------------------
    var cy = null;
    var selectedNodes = new Set();
    var taxonomyData = null;
    var availableRestrictions = null;
    var wsomPollingInterval = null;
    var wsomParentId = null;
    var wsomValidationMode = false;

    // -- Initialisation ------------------------------------------------------
    initCytoscape();
    loadGraph();

    // -- Cytoscape setup -----------------------------------------------------
    function initCytoscape() {
        cy = cytoscape({
            container: document.getElementById("cy"),
            style: [
                {
                    selector: "node",
                    style: {
                        "label": function (ele) {
                            var d = ele.data();
                            var label = d.displayName || d.id.substring(0, 6);
                            return label + "\n(" + d.size + " rows)";
                        },
                        "text-wrap": "wrap",
                        "text-valign": "center",
                        "text-halign": "center",
                        "background-color": function (ele) {
                            return coverageColor(ele.data("coverage"), ele.data("hasChildren"));
                        },
                        "shape": "roundrectangle",
                        "width": "label",
                        "height": "label",
                        "padding": "14px",
                        "font-size": "11px",
                        "border-width": 1,
                        "border-color": "var(--border)",
                        "color": "var(--text)",
                    },
                },
                {
                    selector: "node[?proposed]",
                    style: {
                        "border-width": 2,
                        "border-color": "#f59e0b",
                        "border-style": "dashed",
                    },
                },
                {
                    selector: "node:selected",
                    style: {
                        "border-width": 3,
                        "border-color": "var(--accent)",
                    },
                },
                {
                    selector: "edge",
                    style: {
                        "width": 2,
                        "line-color": "var(--text-muted)",
                        "target-arrow-color": "var(--text-muted)",
                        "target-arrow-shape": "triangle",
                        "curve-style": "bezier",
                    },
                },
            ],
            layout: { name: "preset" },
            boxSelectionEnabled: true,
            selectionType: "additive",
        });

        cy.on("select", "node", onNodeSelect);
        cy.on("unselect", "node", onNodeUnselect);
        cy.on("tap", function (evt) {
            if (evt.target === cy) {
                cy.elements().unselect();
                selectedNodes.clear();
                updateToolbarState();
                showDetailPlaceholder();
            }
        });
    }

    // -- Coverage color ------------------------------------------------------
    function coverageColor(coverage, hasChildren) {
        if (!hasChildren) return "var(--bg-card)";
        var hue = Math.round(coverage * 120);
        return "hsl(" + hue + ", 55%, 85%)";
    }

    // -- Graph loading -------------------------------------------------------
    function loadGraph() {
        fetch("/taxonomy/api/graph")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) return;
                taxonomyData = data;
                renderGraph(data);
            })
            .catch(function (err) {
                console.error("Failed to load taxonomy:", err);
            });
    }

    function renderGraph(data) {
        var elements = [];

        data.nodes.forEach(function (node) {
            var displayName = node.name || "";
            if (!displayName && node.restrictions && node.restrictions.length > 0) {
                displayName = node.restrictions.map(function (r) {
                    return r.column + " " + r.operator + " " + r.value;
                }).join(", ");
            }
            if (!displayName) {
                displayName = node.origin === "root" ? "Root" : node.id.substring(0, 6);
            }

            elements.push({
                group: "nodes",
                data: {
                    id: node.id,
                    name: node.name,
                    displayName: displayName,
                    size: node.size,
                    coverage: node.coverage,
                    hasChildren: node.child_ids.length > 0,
                    origin: node.origin,
                    source_ids: node.source_ids,
                    restrictions: node.restrictions,
                    parent_ids: node.parent_ids,
                    child_ids: node.child_ids,
                    proposed: node.proposed || false,
                },
            });
        });

        data.edges.forEach(function (edge) {
            elements.push({
                group: "edges",
                data: {
                    id: edge.source + "->" + edge.target,
                    source: edge.source,
                    target: edge.target,
                },
            });
        });

        cy.elements().remove();
        cy.add(elements);

        cy.layout({
            name: "dagre",
            rankDir: "TB",
            nodeSep: 60,
            rankSep: 80,
            edgeSep: 20,
            animate: false,
            fit: true,
            padding: 30,
        }).run();

        // Restore selection
        selectedNodes.forEach(function (id) {
            var node = cy.getElementById(id);
            if (node.length) {
                node.select();
            } else {
                selectedNodes.delete(id);
            }
        });

        updateToolbarState();

        // Refresh detail if something is selected
        if (selectedNodes.size === 1) {
            showConceptDetail(Array.from(selectedNodes)[0]);
        } else if (selectedNodes.size === 0) {
            showDetailPlaceholder();
        }
    }

    // -- Select a newly created concept --------------------------------------
    function selectNewConcept(data) {
        taxonomyData = data;
        selectedNodes.clear();
        if (data.new_concept_id) {
            selectedNodes.add(data.new_concept_id);
        }
        renderGraph(data);
    }

    // -- Selection -----------------------------------------------------------
    function onNodeSelect(evt) {
        selectedNodes.add(evt.target.id());
        updateToolbarState();
        if (selectedNodes.size === 1) {
            showConceptDetail(evt.target.id());
        } else {
            showMultiSelectionDetail();
        }
    }

    function onNodeUnselect(evt) {
        selectedNodes.delete(evt.target.id());
        updateToolbarState();
        if (selectedNodes.size === 1) {
            showConceptDetail(Array.from(selectedNodes)[0]);
        } else if (selectedNodes.size === 0) {
            showDetailPlaceholder();
        } else {
            showMultiSelectionDetail();
        }
    }

    // -- Toolbar state -------------------------------------------------------
    function updateToolbarState() {
        if (wsomValidationMode) {
            document.getElementById("btn-add-subconcept").disabled = true;
            document.getElementById("btn-find-subconcepts").disabled = true;
            document.getElementById("btn-complement").disabled = true;
            document.getElementById("btn-union").disabled = true;
            document.getElementById("btn-intersection").disabled = true;
            document.getElementById("btn-find-intersections").disabled = true;
            document.getElementById("btn-merge").disabled = true;
            document.getElementById("btn-delete").disabled = true;
            document.getElementById("btn-reset-taxonomy").disabled = true;
            document.getElementById("btn-export-rdf").disabled = true;
            return;
        }

        var count = selectedNodes.size;
        document.getElementById("btn-add-subconcept").disabled = (count !== 1);
        document.getElementById("btn-find-subconcepts").disabled = (count !== 1);
        document.getElementById("btn-complement").disabled = (count < 1);
        document.getElementById("btn-union").disabled = (count < 2);
        document.getElementById("btn-intersection").disabled = (count < 2);
        document.getElementById("btn-find-intersections").disabled = (count < 2);
        document.getElementById("btn-merge").disabled = (count < 2);
        document.getElementById("btn-reset-taxonomy").disabled = false;

        // Don't allow deleting root
        var canDelete = count > 0;
        if (count > 0 && taxonomyData) {
            selectedNodes.forEach(function (id) {
                if (id === taxonomyData.root_id) canDelete = false;
            });
        }
        document.getElementById("btn-delete").disabled = !canDelete;
        document.getElementById("btn-export-rdf").disabled = !taxonomyData;
    }

    // -- Detail panel --------------------------------------------------------
    function showDetailPlaceholder() {
        document.getElementById("detail-panel").innerHTML =
            '<div class="detail-placeholder"><p class="text-muted">Select a concept to view details</p></div>';
    }

    function showMultiSelectionDetail() {
        var count = selectedNodes.size;
        document.getElementById("detail-panel").innerHTML =
            '<div class="detail-placeholder"><p class="text-muted">' +
            count + ' concepts selected</p></div>';
    }

    function getConceptLabel(conceptId) {
        var node = findNode(conceptId);
        if (!node) return conceptId.substring(0, 6);
        if (node.name) return node.name;
        if (node.restrictions && node.restrictions.length > 0) {
            return node.restrictions.map(function (r) {
                return r.column + " " + r.operator + " " + r.value;
            }).join(", ");
        }
        if (node.origin === "root") return "Root";
        return conceptId.substring(0, 6);
    }

    function buildOriginDescription(node) {
        if (node.origin === "root") return "";

        var sourceNames = (node.source_ids || []).map(function (sid) {
            return escapeHtml(getConceptLabel(sid));
        });

        if (sourceNames.length === 0) return "";

        var label = "";
        switch (node.origin) {
            case "restriction":
                label = "Subconcept of " + sourceNames.join(", ");
                break;
            case "complement":
                label = "Complement of " + sourceNames.join(", ");
                break;
            case "union":
                label = "Union of " + sourceNames.join(", ");
                break;
            case "intersection":
                label = "Intersection of " + sourceNames.join(", ");
                break;
            case "wsom":
                label = "WSOM discovery from " + sourceNames.join(", ");
                break;
            case "merge":
                label = "Merge of " + sourceNames.join(", ");
                break;
            default:
                label = node.origin;
        }

        return '<div class="detail-origin"><h4>Origin</h4><p>' + label + '</p></div>';
    }

    var detailAvailableRestrictions = null;

    function showConceptDetail(conceptId) {
        var node = findNode(conceptId);
        if (!node) return;

        var coveragePct = (node.coverage * 100).toFixed(1);
        var coverageClass = node.child_ids.length === 0 ? "" :
            (node.coverage >= 1.0 ? "coverage-full" : "coverage-partial");

        var originHtml = buildOriginDescription(node);
        var isRoot = (node.origin === "root");

        var restrictionsHtml = "";
        if (!isRoot) {
            restrictionsHtml =
                '<div class="detail-restrictions">' +
                '  <h4>Restrictions</h4>' +
                '  <div id="detail-restrictions-builder"><span class="text-muted">Loading...</span></div>' +
                '  <div class="detail-restrictions-actions">' +
                '    <button class="btn btn-secondary btn-sm" id="btn-detail-add-restriction">+ Add</button>' +
                '    <button class="btn btn-primary btn-sm" id="btn-detail-save-restrictions">Save</button>' +
                '  </div>' +
                '</div>';
        }

        var html =
            '<div class="detail-content">' +
            '  <div class="form-group">' +
            '    <label>Name:</label>' +
            '    <input type="text" class="detail-name-input form-input" id="detail-name" ' +
            '           value="' + escapeHtml(node.name) + '" ' +
            '           placeholder="Unnamed concept" data-id="' + node.id + '">' +
            '  </div>' +
            '  <div class="detail-stats">' +
            '    <span class="stat">' + node.size + ' rows</span>' +
            '    <span class="stat ' + coverageClass + '">' + coveragePct + '% covered</span>' +
            '  </div>' +
            originHtml +
            restrictionsHtml +
            '</div>';

        document.getElementById("detail-panel").innerHTML = html;

        var nameInput = document.getElementById("detail-name");
        nameInput.addEventListener("blur", function () { renameConcept(this); });
        nameInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") this.blur();
        });

        // Load editable restrictions for non-root concepts
        if (!isRoot && node.parent_ids && node.parent_ids.length > 0) {
            var parentId = node.parent_ids[0];
            fetch("/taxonomy/api/concept/" + encodeURIComponent(parentId) + "/restrictions")
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.error) return;
                    detailAvailableRestrictions = data.available;
                    var builder = document.getElementById("detail-restrictions-builder");
                    if (!builder) return;
                    builder.innerHTML = "";

                    var currentRestrictions = node.restrictions || [];
                    if (currentRestrictions.length === 0) {
                        buildRestrictionRow(builder, detailAvailableRestrictions);
                    } else {
                        currentRestrictions.forEach(function (r) {
                            buildRestrictionRow(builder, detailAvailableRestrictions, r);
                        });
                    }

                    var addBtn = document.getElementById("btn-detail-add-restriction");
                    if (addBtn) {
                        addBtn.addEventListener("click", function () {
                            var b = document.getElementById("detail-restrictions-builder");
                            if (b) buildRestrictionRow(b, detailAvailableRestrictions);
                        });
                    }
                    var saveBtn = document.getElementById("btn-detail-save-restrictions");
                    if (saveBtn) {
                        saveBtn.addEventListener("click", function () {
                            saveConceptRestrictions(conceptId);
                        });
                    }
                });
        }
    }

    function saveConceptRestrictions(conceptId) {
        var builder = document.getElementById("detail-restrictions-builder");
        if (!builder) return;
        var restrictions = collectRestrictions(builder);

        fetch("/taxonomy/api/concept/" + encodeURIComponent(conceptId), {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ restrictions: restrictions }),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { alert("Error: " + data.error); return; }
                taxonomyData = data;
                renderGraph(data);
            });
    }

    function findNode(conceptId) {
        if (!taxonomyData) return null;
        for (var i = 0; i < taxonomyData.nodes.length; i++) {
            if (taxonomyData.nodes[i].id === conceptId) return taxonomyData.nodes[i];
        }
        return null;
    }

    // -- Rename --------------------------------------------------------------
    function renameConcept(input) {
        var id = input.dataset.id;
        var newName = input.value.trim();
        fetch("/taxonomy/api/concept/" + encodeURIComponent(id), {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: newName }),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { alert("Error: " + data.error); return; }
                taxonomyData = data;
                renderGraph(data);
            });
    }

    // -- Add Subconcept (Modal) ----------------------------------------------
    document.getElementById("btn-add-subconcept").addEventListener("click", function () {
        if (selectedNodes.size !== 1) return;
        var parentId = Array.from(selectedNodes)[0];
        openAddSubconceptModal(parentId);
    });

    function openAddSubconceptModal(parentId) {
        fetch("/taxonomy/api/concept/" + encodeURIComponent(parentId) + "/restrictions")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { alert("Error: " + data.error); return; }
                availableRestrictions = data.available;
                document.getElementById("subconcept-name").value = "";
                document.getElementById("restrictions-builder").innerHTML = "";
                addRestrictionRow();
                document.getElementById("modal-add-subconcept").hidden = false;
                document.getElementById("modal-add-subconcept").dataset.parentId = parentId;
            });
    }

    // -- Restriction builder (generic) ----------------------------------------
    function buildRestrictionRow(container, available, initial) {
        var rowDiv = document.createElement("div");
        rowDiv.className = "restriction-row";

        // Column select
        var colSelect = document.createElement("select");
        colSelect.className = "form-input restriction-col";
        var defaultOpt = document.createElement("option");
        defaultOpt.value = "";
        defaultOpt.textContent = "-- Column --";
        colSelect.appendChild(defaultOpt);
        available.forEach(function (col) {
            var opt = document.createElement("option");
            opt.value = col.column;
            opt.textContent = col.column + " (" + col.type + ")";
            colSelect.appendChild(opt);
        });

        // Operator select
        var opSelect = document.createElement("select");
        opSelect.className = "form-input restriction-op";
        opSelect.disabled = true;

        // Value container
        var valueContainer = document.createElement("span");
        valueContainer.className = "restriction-value-container";

        // Remove button
        var removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "btn btn-secondary restriction-remove";
        removeBtn.textContent = "\u00D7";
        removeBtn.addEventListener("click", function () { rowDiv.remove(); });

        rowDiv.appendChild(colSelect);
        rowDiv.appendChild(opSelect);
        rowDiv.appendChild(valueContainer);
        rowDiv.appendChild(removeBtn);
        container.appendChild(rowDiv);

        // On column change
        colSelect.addEventListener("change", function () {
            var colName = this.value;
            var colInfo = null;
            for (var i = 0; i < available.length; i++) {
                if (available[i].column === colName) {
                    colInfo = available[i];
                    break;
                }
            }

            opSelect.innerHTML = "";
            valueContainer.innerHTML = "";

            if (!colInfo) {
                opSelect.disabled = true;
                return;
            }

            // Populate operators
            opSelect.disabled = false;
            colInfo.operators.forEach(function (op) {
                var opt = document.createElement("option");
                opt.value = op;
                opt.textContent = op;
                opSelect.appendChild(opt);
            });

            // Populate value input
            if (colInfo.type === "categorical") {
                var valSelect = document.createElement("select");
                valSelect.className = "form-input restriction-val";
                colInfo.values.forEach(function (v) {
                    var opt = document.createElement("option");
                    opt.value = v;
                    opt.textContent = v;
                    valSelect.appendChild(opt);
                });
                valueContainer.appendChild(valSelect);
            } else if (colInfo.type === "numeric") {
                var valInput = document.createElement("input");
                valInput.type = "number";
                valInput.className = "form-input restriction-val";
                valInput.step = "any";
                valInput.placeholder = colInfo.values.min + " \u2013 " + colInfo.values.max;
                valueContainer.appendChild(valInput);
            } else if (colInfo.type === "date") {
                var valInput = document.createElement("input");
                valInput.type = "date";
                valInput.className = "form-input restriction-val";
                valInput.min = colInfo.values.min;
                valInput.max = colInfo.values.max;
                valueContainer.appendChild(valInput);
            }
        });

        // Pre-populate if initial values provided
        if (initial) {
            colSelect.value = initial.column;
            colSelect.dispatchEvent(new Event("change"));
            opSelect.value = initial.operator;
            var valEl = valueContainer.querySelector(".restriction-val");
            if (valEl) {
                valEl.value = String(initial.value);
            }
        }
    }

    function collectRestrictions(container) {
        var restrictions = [];
        var rows = container.querySelectorAll(".restriction-row");
        for (var i = 0; i < rows.length; i++) {
            var row = rows[i];
            var col = row.querySelector(".restriction-col").value;
            var opEl = row.querySelector(".restriction-op");
            var op = opEl ? opEl.value : "";
            var valEl = row.querySelector(".restriction-val");
            var val = valEl ? valEl.value : "";
            if (col && op && val !== "") {
                restrictions.push({ column: col, operator: op, value: val });
            }
        }
        return restrictions;
    }

    // -- Add Subconcept modal restriction rows --------------------------------
    function addRestrictionRow() {
        var builder = document.getElementById("restrictions-builder");
        buildRestrictionRow(builder, availableRestrictions);
    }

    document.getElementById("btn-add-restriction-row").addEventListener("click", function () {
        addRestrictionRow();
    });

    // -- Confirm add subconcept ----------------------------------------------
    document.getElementById("btn-confirm-subconcept").addEventListener("click", function () {
        var modal = document.getElementById("modal-add-subconcept");
        var parentId = modal.dataset.parentId;
        var name = document.getElementById("subconcept-name").value.trim();
        var restrictions = collectRestrictions(document.getElementById("restrictions-builder"));

        if (restrictions.length === 0) {
            alert("Add at least one restriction.");
            return;
        }

        fetch("/taxonomy/api/concept", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                parent_id: parentId,
                restrictions: restrictions,
                name: name,
            }),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { alert("Error: " + data.error); return; }
                modal.hidden = true;
                selectNewConcept(data);
            });
    });

    // -- Complement ----------------------------------------------------------
    document.getElementById("btn-complement").addEventListener("click", function () {
        var ids = Array.from(selectedNodes);
        fetch("/taxonomy/api/complement", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ concept_ids: ids }),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { alert("Error: " + data.error); return; }
                selectNewConcept(data);
            });
    });

    // -- Union ---------------------------------------------------------------
    document.getElementById("btn-union").addEventListener("click", function () {
        var ids = Array.from(selectedNodes);
        fetch("/taxonomy/api/union", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ concept_ids: ids }),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { alert("Error: " + data.error); return; }
                selectNewConcept(data);
            });
    });

    // -- Intersection --------------------------------------------------------
    document.getElementById("btn-intersection").addEventListener("click", function () {
        var ids = Array.from(selectedNodes);
        fetch("/taxonomy/api/intersection", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ concept_ids: ids }),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { alert("Error: " + data.error); return; }
                selectNewConcept(data);
            });
    });

    // -- Find Intersections --------------------------------------------------
    document.getElementById("btn-find-intersections").addEventListener("click", function () {
        var ids = Array.from(selectedNodes);
        fetch("/taxonomy/api/find-intersections", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ concept_ids: ids }),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { alert(data.error); return; }
                taxonomyData = data;
                selectedNodes.clear();
                renderGraph(data);
            });
    });

    // -- Merge ---------------------------------------------------------------
    document.getElementById("btn-merge").addEventListener("click", function () {
        var ids = Array.from(selectedNodes);
        fetch("/taxonomy/api/merge", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ concept_ids: ids }),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { alert("Error: " + data.error); return; }
                selectNewConcept(data);
            });
    });

    // -- Delete --------------------------------------------------------------
    document.getElementById("btn-delete").addEventListener("click", function () {
        var ids = Array.from(selectedNodes);
        if (!confirm("Delete " + ids.length + " concept(s) and their descendants?")) return;

        var chain = Promise.resolve();
        ids.forEach(function (id) {
            chain = chain.then(function () {
                return fetch("/taxonomy/api/concept/" + encodeURIComponent(id), {
                    method: "DELETE",
                }).then(function (r) { return r.json(); });
            });
        });

        chain.then(function (data) {
            selectedNodes.clear();
            if (data && !data.error) {
                taxonomyData = data;
                renderGraph(data);
            } else {
                loadGraph();
            }
            showDetailPlaceholder();
        });
    });

    // -- Reset ---------------------------------------------------------------
    document.getElementById("btn-reset-taxonomy").addEventListener("click", function () {
        if (!confirm("Reset taxonomy? This will remove all concepts.")) return;
        fetch("/taxonomy/api/reset", { method: "POST" })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { alert("Error: " + data.error); return; }
                selectedNodes.clear();
                taxonomyData = data;
                renderGraph(data);
                showDetailPlaceholder();
            });
    });

    // -- WSOM discovery ------------------------------------------------------
    document.getElementById("btn-find-subconcepts").addEventListener("click", function () {
        if (selectedNodes.size !== 1) return;
        wsomParentId = Array.from(selectedNodes)[0];

        // Set default epochs based on concept size
        var conceptNode = taxonomyData && taxonomyData.nodes.find(function (n) { return n.id === wsomParentId; });
        if (conceptNode && conceptNode.size > 0) {
            var defaultEpochs = Math.max(5, Math.min(5000, Math.round(250000 / conceptNode.size)));
            document.getElementById("wsom-epochs").value = defaultEpochs;
        }

        // Populate column checkboxes
        var container = document.getElementById("wsom-ignore-columns");
        container.innerHTML = '<span class="text-muted">Loading columns...</span>';
        fetch("/taxonomy/api/columns")
            .then(function (r) { return r.json(); })
            .then(function (columns) {
                container.innerHTML = "";
                columns.forEach(function (col) {
                    var label = document.createElement("label");
                    label.className = "wsom-column-checkbox";
                    var cb = document.createElement("input");
                    cb.type = "checkbox";
                    cb.value = col.name;
                    label.appendChild(cb);
                    label.appendChild(document.createTextNode(" " + col.name + " (" + col.type + ")"));
                    container.appendChild(label);
                });
            });

        document.getElementById("modal-wsom-params").hidden = false;
    });

    document.getElementById("btn-start-wsom").addEventListener("click", function () {
        var mapSize = parseInt(document.getElementById("wsom-map-size").value, 10);
        var sparcity = parseFloat(document.getElementById("wsom-sparcity").value);
        var epochs = parseInt(document.getElementById("wsom-epochs").value, 10);
        var mergeThreshold = parseFloat(document.getElementById("wsom-merge-threshold").value) / 100;

        var ignoreColumns = [];
        document.querySelectorAll("#wsom-ignore-columns input:checked").forEach(function (cb) {
            ignoreColumns.push(cb.value);
        });

        document.getElementById("modal-wsom-params").hidden = true;
        document.getElementById("wsom-progress-overlay").hidden = false;
        document.getElementById("wsom-progress-bar").style.width = "0%";
        document.getElementById("wsom-progress-text").textContent = "Epoch 0 / " + epochs;

        fetch("/taxonomy/api/wsom/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                concept_id: wsomParentId,
                map_size: mapSize,
                sparcity_coeff: sparcity,
                n_epochs: epochs,
                merge_f1_threshold: mergeThreshold,
                ignore_columns: ignoreColumns,
            }),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) {
                    document.getElementById("wsom-progress-overlay").hidden = true;
                    alert("Error: " + data.error);
                    return;
                }
                startWsomPolling();
            })
            .catch(function (err) {
                document.getElementById("wsom-progress-overlay").hidden = true;
                alert("Error starting training: " + err);
            });
    });

    function startWsomPolling() {
        wsomPollingInterval = setInterval(function () {
            fetch("/taxonomy/api/wsom/progress")
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.status === "training") {
                        var pct = (data.progress / data.total * 100).toFixed(1);
                        document.getElementById("wsom-progress-bar").style.width = pct + "%";
                        document.getElementById("wsom-progress-text").textContent =
                            "Epoch " + data.progress + " / " + data.total;
                    } else if (data.status === "complete") {
                        clearInterval(wsomPollingInterval);
                        wsomPollingInterval = null;
                        document.getElementById("wsom-progress-overlay").hidden = true;

                        if (data.proposed_ids.length === 0) {
                            alert("No characteristic sub-concepts were found. Try adjusting the parameters.");
                            return;
                        }

                        wsomValidationMode = true;
                        taxonomyData = data.graph;
                        selectedNodes.clear();
                        renderGraph(data.graph);
                        document.getElementById("wsom-validation-bar").hidden = false;
                        if (cy) cy.resize();
                        updateToolbarState();
                    } else if (data.status === "error") {
                        clearInterval(wsomPollingInterval);
                        wsomPollingInterval = null;
                        document.getElementById("wsom-progress-overlay").hidden = true;
                        alert("Training error: " + data.error);
                    } else if (data.status === "idle") {
                        clearInterval(wsomPollingInterval);
                        wsomPollingInterval = null;
                        document.getElementById("wsom-progress-overlay").hidden = true;
                    }
                })
                .catch(function () {
                    // Ignore transient fetch errors during polling
                });
        }, 500);
    }

    document.getElementById("btn-wsom-validate").addEventListener("click", function () {
        resolveWsom("validate");
    });

    document.getElementById("btn-wsom-cancel").addEventListener("click", function () {
        resolveWsom("cancel");
    });

    document.getElementById("btn-wsom-retry").addEventListener("click", function () {
        resolveWsom("retry");
    });

    function resolveWsom(action) {
        fetch("/taxonomy/api/wsom/resolve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: action }),
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { alert("Error: " + data.error); return; }

                wsomValidationMode = false;
                document.getElementById("wsom-validation-bar").hidden = true;

                taxonomyData = data;
                selectedNodes.clear();
                renderGraph(data);
                if (cy) cy.resize();
                updateToolbarState();

                if (action === "retry" && data.retry_parent_id) {
                    wsomParentId = data.retry_parent_id;
                    selectedNodes.add(wsomParentId);
                    var node = cy.getElementById(wsomParentId);
                    if (node.length) node.select();
                    document.getElementById("modal-wsom-params").hidden = false;
                }
            });
    }

    // -- Export RDF ----------------------------------------------------------
    document.getElementById("btn-export-rdf").addEventListener("click", function () {
        var stem = "data";
        if (typeof uploadedFilename === "string" && uploadedFilename) {
            stem = uploadedFilename.replace(/\.[^/.]+$/, "");
        }
        document.getElementById("rdf-concept-ns").value =
            "http://example.org/" + encodeURIComponent(stem) + "/schema#";
        document.getElementById("rdf-entity-ns").value =
            "http://example.org/" + encodeURIComponent(stem) + "/data/";
        document.getElementById("modal-export-rdf").hidden = false;
    });

    document.getElementById("btn-confirm-export-rdf").addEventListener("click", function () {
        var conceptNs = document.getElementById("rdf-concept-ns").value.trim();
        var entityNs = document.getElementById("rdf-entity-ns").value.trim();
        if (!conceptNs || !entityNs) {
            alert("Please provide both namespace URIs.");
            return;
        }

        var btn = document.getElementById("btn-confirm-export-rdf");
        btn.disabled = true;
        btn.textContent = "Exporting...";

        fetch("/taxonomy/api/export-rdf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                concept_namespace: conceptNs,
                entity_namespace: entityNs,
            }),
        })
            .then(function (r) {
                if (!r.ok) throw new Error("Export failed: " + r.statusText);
                return r.blob();
            })
            .then(function (blob) {
                var stem = "taxonomy";
                if (typeof uploadedFilename === "string" && uploadedFilename) {
                    stem = uploadedFilename.replace(/\.[^/.]+$/, "");
                }
                var a = document.createElement("a");
                a.href = URL.createObjectURL(blob);
                a.download = stem + ".ttl";
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);
                document.getElementById("modal-export-rdf").hidden = true;
            })
            .catch(function (err) {
                alert("Export error: " + err.message);
            })
            .finally(function () {
                btn.disabled = false;
                btn.textContent = "Download .ttl";
            });
    });

    // -- Modal close ---------------------------------------------------------
    document.querySelectorAll("[data-dismiss='modal']").forEach(function (btn) {
        btn.addEventListener("click", function () {
            this.closest(".modal-overlay").hidden = true;
        });
    });

    // -- Utility -------------------------------------------------------------
    function escapeHtml(str) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }
});
