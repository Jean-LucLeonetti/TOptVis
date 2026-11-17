const el=id=>document.getElementById(id);
const format=value=>value===0 ? "0" : value<.001 || value>1e4 ? value.toExponential(1) : value.toPrecision(3);
export function createTrainingChart(prefix) {
  const panel=el(`${prefix}-learning-panel`), chart=el(`${prefix}-learning-chart`);
  let history=[],model=null;
  function draw() {
    panel.hidden=!history.length;
    if(!history.length) {chart.replaceChildren();return;}
    const width=Math.max(320,chart.clientWidth || 720),height=250,left=80,right=width-24,top=28,bottom=height-48;
    const last=history.at(-1), maximum=Math.max(1e-15,...history.flatMap(row=>[row.train_loss,row.validation_loss]));
    const maxEpoch=Math.max(1,last.epoch),x=epoch=>left+(right-left)*epoch/maxEpoch;
    const y=value=>bottom-(bottom-top)*value/maximum;
    chart.setAttribute("viewBox",`0 0 ${width} ${height}`);
    let svg=`<title>Training and validation mean squared error by epoch</title>`;
    for(let i=0;i<=4;i++) {
      const value=maximum*i/4,epoch=maxEpoch*i/4;
      svg+=`<path d="M${left} ${y(value)}H${right}" stroke="#e4e9ed"/><text x="${left-10}" y="${y(value)+5}" text-anchor="end">${format(value)}</text><text x="${x(epoch)}" y="${bottom+23}" text-anchor="middle">${Math.round(epoch)}</text>`;
    }
    for(const [key,color] of [["train_loss","#196964"],["validation_loss","#4b7eaf"]]) {
      const path=history.map((row,i)=>`${i ? "L" : "M"}${x(row.epoch)},${y(row[key])}`).join("");
      svg+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
    }
    if(model) svg+=`<path d="M${x(model.best_epoch)} ${top}V${bottom}" stroke="#7961b2" stroke-dasharray="5 4"/>`;
    svg+=`<text x="${(left+right)/2}" y="${height-5}" text-anchor="middle">Epoch</text><text x="${left}" y="18">Mean squared error</text>`;
    chart.innerHTML=svg;
    const best=model ? ` Using epoch ${model.best_epoch}, selected by validation loss.` : "";
    el(`${prefix}-learning-note`).textContent=`Epoch ${last.epoch}. Training ${format(last.train_loss)} / validation ${format(last.validation_loss)}.${best}`;
    chart.setAttribute("aria-label",`Training and validation loss through epoch ${last.epoch}.${best}`);
  }
  new ResizeObserver(draw).observe(chart.parentElement);
  return {update(rows, fitted=null) {history=rows || [];model=fitted;draw();}};
}
