document.addEventListener("DOMContentLoaded", function () {
    // -- State ---------------------------------------------------------------
    var cy = null;
    var selectedNodes = new Set();
    var taxonomyData = null;
    var availableRestrictions = null;

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
        var count = selectedNodes.size;
        document.getElementById("btn-add-subconcept").disabled = (count !== 1);
        document.getElementById("btn-complement").disabled = (count < 1);
        document.getElementById("btn-union").disabled = (count < 2);
        document.getElementById("btn-intersection").disabled = (count < 2);

        // Don't allow deleting root
        var canDelete = count > 0;
        if (count > 0 && taxonomyData) {
            selectedNodes.forEach(function (id) {
                if (id === taxonomyData.root_id) canDelete = false;
            });
        }
        document.getElementById("btn-delete").disabled = !canDelete;
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
            default:
                label = node.origin;
        }

        return '<div class="detail-origin"><h4>Origin</h4><p>' + label + '</p></div>';
    }

    function showConceptDetail(conceptId) {
        var node = findNode(conceptId);
        if (!node) return;

        var restrictionsHtml = "";
        if (node.restrictions && node.restrictions.length > 0) {
            restrictionsHtml = '<div class="detail-restrictions"><h4>Restrictions</h4><ul>';
            node.restrictions.forEach(function (r) {
                restrictionsHtml += "<li><strong>" + escapeHtml(r.column) +
                    "</strong> " + escapeHtml(r.operator) +
                    " " + escapeHtml(String(r.value)) + "</li>";
            });
            restrictionsHtml += "</ul></div>";
        }

        var coveragePct = (node.coverage * 100).toFixed(1);
        var coverageClass = node.child_ids.length === 0 ? "" :
            (node.coverage >= 1.0 ? "coverage-full" : "coverage-partial");

        var originHtml = buildOriginDescription(node);

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

    // -- Restriction builder -------------------------------------------------
    function addRestrictionRow() {
        var builder = document.getElementById("restrictions-builder");
        var rowDiv = document.createElement("div");
        rowDiv.className = "restriction-row";

        // Column select
        var colSelect = document.createElement("select");
        colSelect.className = "form-input restriction-col";
        var defaultOpt = document.createElement("option");
        defaultOpt.value = "";
        defaultOpt.textContent = "-- Column --";
        colSelect.appendChild(defaultOpt);
        availableRestrictions.forEach(function (col) {
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
        builder.appendChild(rowDiv);

        // On column change
        colSelect.addEventListener("change", function () {
            var colName = this.value;
            var colInfo = null;
            for (var i = 0; i < availableRestrictions.length; i++) {
                if (availableRestrictions[i].column === colName) {
                    colInfo = availableRestrictions[i];
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
    }

    document.getElementById("btn-add-restriction-row").addEventListener("click", function () {
        addRestrictionRow();
    });

    // -- Confirm add subconcept ----------------------------------------------
    document.getElementById("btn-confirm-subconcept").addEventListener("click", function () {
        var modal = document.getElementById("modal-add-subconcept");
        var parentId = modal.dataset.parentId;
        var name = document.getElementById("subconcept-name").value.trim();
        var restrictions = [];

        var rows = document.querySelectorAll("#restrictions-builder .restriction-row");
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
