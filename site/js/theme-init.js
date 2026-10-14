// Apply the saved or system color theme before first paint.
(function () {
  var t = null;
  try { t = localStorage.getItem("srag-theme"); } catch (e) { /* storage blocked */ }
  if (t !== "light" && t !== "dark") t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.dataset.theme = t;
})();
