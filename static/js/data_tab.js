document.addEventListener("DOMContentLoaded", function () {
    var container = document.getElementById("columns-container");
    if (!container) return;

    container.addEventListener("change", function (event) {
        var target = event.target;

        if (target.classList.contains("col-type-select")) {
            var columnName = target.dataset.column;
            updateColumn(columnName, { user_type: target.value });
        }

        if (target.classList.contains("col-include-check")) {
            var columnName = target.dataset.column;
            updateColumn(columnName, { include: target.checked });
        }
    });
});

function updateColumn(columnName, data) {
    fetch("/data/columns/" + encodeURIComponent(columnName), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    })
        .then(function (response) {
            return response.json();
        })
        .then(function (result) {
            if (result.error) {
                alert("Error: " + result.error);
                return;
            }
            updateColumnCard(result.column);
            var countEl = document.getElementById("encoded-col-count");
            if (countEl) {
                countEl.textContent = result.encoded_col_count;
            }
        })
        .catch(function (err) {
            console.error("Failed to update column:", err);
        });
}

function updateColumnCard(columnData) {
    // Find the card by data-column attribute
    var cards = document.querySelectorAll(".column-card");
    var card = null;
    for (var i = 0; i < cards.length; i++) {
        if (cards[i].dataset.column === columnData.column_name) {
            card = cards[i];
            break;
        }
    }
    if (!card) return;

    // Update summary content
    var summaryDiv = card.querySelector(".summary-content");
    if (summaryDiv && columnData.summary_html) {
        summaryDiv.innerHTML = columnData.summary_html;
    }

    // Update select value
    var select = card.querySelector(".col-type-select");
    if (select) {
        select.value = columnData.user_type;
    }

    // Update checkbox
    var checkbox = card.querySelector(".col-include-check");
    if (checkbox) {
        checkbox.checked = columnData.include;
    }
}
