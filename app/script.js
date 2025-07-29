function openTab(evt, tabName) {
  const tabContents = document.getElementsByClassName("tab-content");
  for (let i = 0; i < tabContents.length; i++) {
    tabContents[i].classList.remove("active");
  }

  const tabs = document.getElementsByClassName("tab");
  for (let i = 0; i < tabs.length; i++) {
    tabs[i].classList.remove("active");
  }

  document.getElementById(tabName).classList.add("active");
  evt.currentTarget.classList.add("active");
}

$("#second-per-data")[0].addEventListener("input", function () {
  $("#second-per-data-label")[0].textContent =
    "Second Per Data : " + this.value;
});

$("#stride")[0].addEventListener("input", function () {
  $("#stride-label")[0].textContent = "Stride : " + this.value;
});

function runDetection() {
  dataset_path = $("#dataset-path")[0].value;
  split = $("#detection-split")[0].value;
  second_per_data = parseInt($("#second-per-data")[0].value);
  stride = parseInt($("#stride")[0].value);

  check_pointpillar = $("#detection-det-pp")[0].checked;
  check_pv_rcnn = $("#detection-det-pv-rcnn")[0].checked;

  fetch("/detection", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      dataset_path: dataset_path,
      split: split,
      second_per_data: second_per_data,
      stride: stride,
    }),
  });
}

function runTracking() {
  split = $("#tracking-split")[0].value;

  check_pointpillar = $("#tracking-det-pp")[0].checked;
  check_pv_rcnn = $("#tracking-det-pv-rcnn")[0].checked;

  check_mctrack_online = $("#tracking-fw-mctrack-online")[0].checked;
  check_mctrack_global = $("#tracking-fw-mctrack-global")[0].checked;
  check_wu_online = $("#tracking-fw-wu-online")[0].checked;
  check_wu_global = $("#tracking-fw-wu-global")[0].checked;

  fetch("/tracking", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      split: split,
      check_pointpillar: check_pointpillar,
      check_pv_rcnn: check_pv_rcnn,
      check_mctrack_online: check_mctrack_online,
      check_mctrack_global: check_mctrack_global,
      check_wu_online: check_wu_online,
      check_wu_global: check_wu_global,
    }),
  });
}

function runTestPrioritization() {
  split = $("#testing-split")[0].value;

  check_pointpillar = $("#testing-det-pp")[0].checked;
  check_pv_rcnn = $("#testing-det-pv-rcnn")[0].checked;

  check_mctrack_online = $("#testing-fw-mctrack-online")[0].checked;
  check_mctrack_global = $("#testing-fw-mctrack-global")[0].checked;
  check_wu_online = $("#testing-fw-wu-online")[0].checked;
  check_wu_global = $("#testing-fw-wu-global")[0].checked;

  fetch("/testing", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      split: split,
      check_pointpillar: check_pointpillar,
      check_pv_rcnn: check_pv_rcnn,
      check_mctrack_online: check_mctrack_online,
      check_mctrack_global: check_mctrack_global,
      check_wu_online: check_wu_online,
      check_wu_global: check_wu_global,
    }),
  });
}
