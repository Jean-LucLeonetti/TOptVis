import "/static/cantilever.js";
import "/static/rom-training.js";
import "/static/regression.js";

// Workspace navigation stays independent of each workspace's controls and state.
const tabs = Array.from(document.querySelectorAll('.workspace-tabs [role="tab"]'));
const workspaces = new Map(tabs.map(tab => [tab.dataset.workspace, tab]));
const methodTabs = Array.from(document.querySelectorAll('.ml-method-tabs [role="tab"]'));
const methods = new Map(methodTabs.map(tab => [tab.dataset.method, tab]));
let activeMethod = methodTabs[0];

function selectWorkspace(route, {focus = false, updateHistory = false, methodFocus = false} = {}) {
  // Keep links to the original standalone ROM workspace working.
  const [name, method] = (route === "rom-training" ? "ml/rom" : route).split("/");
  const active = workspaces.get(name) || tabs[0];
  activeMethod = methods.get(method) || activeMethod;
  for (const tab of tabs) {
    const selected = tab === active;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !selected;
  }
  for (const tab of methodTabs) {
    const selected = tab === activeMethod;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !selected;
  }
  const isML = active.dataset.workspace === "ml";
  const label = `${active.textContent.trim()}${isML ? ` · ${activeMethod.textContent.trim()}` : ""}`;
  document.title = `TOptVis · ${label}`;
  if (focus) (methodFocus ? activeMethod : active).focus();
  const hash = `#${active.dataset.workspace}${isML ? `/${activeMethod.dataset.method}` : ""}`;
  if (updateHistory && location.hash !== hash) {
    history.pushState(null, "", hash);
  }
  document.dispatchEvent(new CustomEvent("workspacechange", {detail: active.dataset.workspace}));
  if (isML) document.dispatchEvent(new CustomEvent("mlmethodchange", {detail: activeMethod.dataset.method}));
}

function bindTabs(group, routeFor, methodFocus = false) {
  for (const [index, tab] of group.entries()) {
    tab.addEventListener("click", () => selectWorkspace(routeFor(tab), {updateHistory: true}));
    tab.addEventListener("keydown", event => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % group.length;
      if (event.key === "ArrowLeft") next = (index + group.length - 1) % group.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = group.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        selectWorkspace(routeFor(group[next]), {focus: true, updateHistory: true, methodFocus});
      }
    });
  }
}
bindTabs(tabs, tab => tab.dataset.workspace);
bindTabs(methodTabs, tab => `ml/${tab.dataset.method}`, true);

window.addEventListener("hashchange", () => selectWorkspace(location.hash.slice(1)));
selectWorkspace(location.hash.slice(1));
