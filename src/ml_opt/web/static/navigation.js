// Workspace navigation stays independent of each workspace's controls and state.
const tabs = Array.from(document.querySelectorAll('.workspace-tabs [role="tab"]'));
const workspaces = new Map(tabs.map(tab => [tab.dataset.workspace, tab]));
const methodTabs = Array.from(document.querySelectorAll('.ml-method-tabs [role="tab"]'));
const methods = new Map(methodTabs.map(tab => [tab.dataset.method, tab]));
let activeMethod = methodTabs[0];
let regressionView="compliance";
const regressionTab=document.getElementById("tab-regression");
const regressionMenu=document.getElementById("regression-menu");
const regressionItems=Array.from(regressionMenu.querySelectorAll("[data-regression-view]"));

function closeMenu(focus=false) {
  regressionMenu.hidden=true;regressionTab.setAttribute("aria-expanded","false");
  if(focus) regressionTab.focus();
}
function openMenu() {
  regressionMenu.hidden=false;regressionTab.setAttribute("aria-expanded","true");
  regressionItems.find(item=>item.dataset.regressionView===regressionView).focus();
}

function selectWorkspace(route, {focus = false, updateHistory = false, methodFocus = false} = {}) {
  // Keep links to the original standalone ROM workspace working.
  const [name, method, view] = (route === "rom-training" ? "ml/rom" : route).split("/");
  if(method==="regression") regressionView=view==="topology" ? "topology" : "compliance";
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
  const methodLabel=activeMethod.dataset.method==="regression" ? `Regression · ${regressionView==="topology" ? "Topology" : "Compliance"}` : activeMethod.textContent.trim();
  const label = `${active.textContent.trim()}${isML ? ` · ${methodLabel}` : ""}`;
  document.title = `TOptVis · ${label}`;
  if (focus) (methodFocus ? activeMethod : active).focus();
  const hash = `#${active.dataset.workspace}${isML ? `/${activeMethod.dataset.method}${activeMethod.dataset.method==="regression" ? `/${regressionView}` : ""}` : ""}`;
  if (updateHistory && location.hash !== hash) {
    history.pushState(null, "", hash);
  }
  document.dispatchEvent(new CustomEvent("workspacechange", {detail: active.dataset.workspace}));
  for(const item of regressionItems) item.setAttribute("aria-checked",String(item.dataset.regressionView===regressionView));
  document.dispatchEvent(new CustomEvent("regressionviewchange", {detail:regressionView}));
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
        closeMenu();
        selectWorkspace(routeFor(group[next]), {focus: true, updateHistory: true, methodFocus});
      }
    });
  }
}
bindTabs(tabs, tab => tab.dataset.workspace);
for(const tab of methodTabs.filter(tab=>tab!==regressionTab)) tab.addEventListener("click",()=>{
  closeMenu();selectWorkspace(`ml/${tab.dataset.method}`,{updateHistory:true});
});
// Keep arrow-key navigation across both method tabs while Regression opens a menu.
for(const tab of methodTabs) tab.addEventListener("keydown",event=>{
  const index=methodTabs.indexOf(tab);
  const next=event.key==="ArrowRight" ? (index+1)%methodTabs.length : event.key==="ArrowLeft" ?
    (index+methodTabs.length-1)%methodTabs.length : event.key==="Home" ? 0 : event.key==="End" ? methodTabs.length-1 : null;
  if(next!==null) {
    event.preventDefault();closeMenu();
    const target=methodTabs[next];
    selectWorkspace(`ml/${target.dataset.method}${target===regressionTab ? `/${regressionView}` : ""}`,{focus:true,methodFocus:true,updateHistory:true});
  }
});
regressionTab.addEventListener("click",()=>{
  const expanded=!regressionMenu.hidden;
  selectWorkspace(`ml/regression/${regressionView}`,{updateHistory:true});
  if(expanded) closeMenu();else openMenu();
});
regressionTab.addEventListener("keydown",event=>{
  if(event.key==="ArrowDown" || event.key==="ArrowUp") {event.preventDefault();openMenu();}
});
for(const [index,item] of regressionItems.entries()) {
  item.addEventListener("click",()=>{
    selectWorkspace(`ml/regression/${item.dataset.regressionView}`,{updateHistory:true});closeMenu(true);
  });
  item.addEventListener("keydown",event=>{
    if(event.key==="Escape") {event.preventDefault();closeMenu(true);return;}
    if(event.key==="Tab") {closeMenu(true);return;}
    const next=event.key==="ArrowDown" ? (index+1)%regressionItems.length : event.key==="ArrowUp" ?
      (index+regressionItems.length-1)%regressionItems.length : event.key==="Home" ? 0 : event.key==="End" ? regressionItems.length-1 : null;
    if(next!==null) {event.preventDefault();regressionItems[next].focus();}
  });
}
document.addEventListener("click",event=>{if(!event.target.closest(".ml-method-dropdown")) closeMenu();});

window.addEventListener("hashchange", () => {closeMenu();selectWorkspace(location.hash.slice(1));});
selectWorkspace(location.hash.slice(1));
