import "/static/cantilever.js";
import "/static/rom-training.js";

// Workspace navigation stays independent of each workspace's controls and state.
const tabs = Array.from(document.querySelectorAll('.workspace-tabs [role="tab"]'));
const workspaces = new Map(tabs.map(tab => [tab.dataset.workspace, tab]));

function selectWorkspace(name, {focus = false, updateHistory = false} = {}) {
  const active = workspaces.get(name) || tabs[0];
  for (const tab of tabs) {
    const selected = tab === active;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !selected;
  }
  document.title = `TOptVis · ${active.textContent.trim()}`;
  document.getElementById("workspace-label").textContent = `TOptVis · ${active.textContent.trim()}`;
  if (focus) active.focus();
  if (updateHistory && location.hash !== `#${active.dataset.workspace}`) {
    history.pushState(null, "", `#${active.dataset.workspace}`);
  }
  document.dispatchEvent(new CustomEvent("workspacechange", {detail: active.dataset.workspace}));
}

for (const [index, tab] of tabs.entries()) {
  tab.addEventListener("click", () => selectWorkspace(tab.dataset.workspace, {updateHistory: true}));
  tab.addEventListener("keydown", event => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      selectWorkspace(tabs[next].dataset.workspace, {focus: true, updateHistory: true});
    }
  });
}

window.addEventListener("hashchange", () => selectWorkspace(location.hash.slice(1)));
selectWorkspace(location.hash.slice(1));
