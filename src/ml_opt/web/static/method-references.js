const methods = [
  {name:"SIMP",topic:"Material interpolation",description:"A power law reduces the stiffness assigned to intermediate material densities. The implementation uses a small positive void modulus.",paper:"Sigmund (2001), A 99 line topology optimization code written in Matlab",url:"https://doi.org/10.1007/s001580050176"},
  {name:"RAMP",topic:"Material interpolation",description:"Rational interpolation provides an alternative relationship between density and stiffness. Its penalty parameter has a different meaning from the SIMP exponent.",paper:"Stolpe and Svanberg (2001), An alternative interpolation scheme for minimum compliance topology optimization",url:"https://doi.org/10.1007/s001580100129"},
  {name:"Helmholtz filtering",topic:"Regularisation",description:"Smooths the design before material interpolation. TOptVis uses a cell-centred discretisation with zero normal flux; the filter length is a smoothing reference, not a guaranteed member thickness.",paper:"Lazarov and Sigmund (2011), Filters in topology optimization based on Helmholtz-type differential equations",url:"https://doi.org/10.1002/nme.3072"},
  {name:"Projection and robust design",topic:"Regularisation",description:"Smooth thresholding sharpens the material distribution. The robust formulation evaluates eroded, nominal, and dilated fields, minimises their worst objective, and constrains dilated volume. Sharpness continuation changes the approximation during optimisation.",paper:"Wang, Lazarov and Sigmund (2011), On projection methods, convergence and robust formulations in topology optimization",url:"https://doi.org/10.1007/s00158-010-0602-y"},
  {name:"Maximum feature sizes",topic:"Geometric constraints",description:"Relaxed morphological tests constrain solid and void region sizes. These are discretised constraints with tolerances, rather than exact geometric certificates.",paper:"Lazarov and Wang (2017), Maximum length scale in density based topology optimization",url:"https://doi.org/10.1016/j.cma.2017.02.018"},
  {name:"MMA",topic:"Optimisation",description:"Builds and solves a local convex approximation using moving asymptotes. Move limits control how much the design can change at each outer iteration.",paper:"Svanberg (1987), The method of moving asymptotes - a new method for structural optimization",url:"https://doi.org/10.1002/nme.1620240207"},
  {name:"GCMMA",topic:"Optimisation",description:"Checks whether trial approximations are conservative and retries when necessary. This can require several state evaluations per outer iteration. Global convergence describes the algorithm's convergence theory, not a guarantee of finding a global optimum.",paper:"Svanberg (2002), A Class of Globally Convergent Optimization Methods Based on Conservative Convex Separable Approximations",url:"https://doi.org/10.1137/S1052623499362822"},
  {name:"Independent load cases",topic:"Objectives",description:"Forces within a case act together. Different cases are solved independently against the same topology. The weighted mean uses normalised positive weights; worst case uses the largest case objective and ignores weights. Their gradients follow the same aggregation.",paper:"Sigmund (2001), Educational formulation and multiple-load extensions",url:"https://www.topopt.mek.dtu.dk/apps-and-software/a-99-line-topology-optimization-code-written-in-matlab"},
  {name:"Prescribed motion",topic:"Objectives",description:"Supports are shared across cases. Nonzero prescribed motion uses total elastic work, including support reactions, with an adjoint sensitivity. Under imposed motion, minimising this work can favour a softer structure; it is not equivalent to force-only stiffness maximisation.",paper:null,url:null},
  {name:"POD and Galerkin reduction",topic:"Reduced models",description:"POD compresses displacement snapshots into an orthonormal basis. Galerkin projection solves equilibrium in that basis. TOptVis uses a frozen, uncentred basis with optional snapshot normalisation; accuracy depends on training coverage and must be checked on held-out cases.",paper:"Sirovich (1987), Turbulence and the dynamics of coherent structures. I. Coherent structures",url:"https://doi.org/10.1090/qam/910462"},
];

export function initialiseReferences() {
  const dialog = document.getElementById("references-dialog"), list = document.getElementById("references-list");
  for (const method of methods) {
    const article = document.createElement("article"), title = document.createElement("h3"), text = document.createElement("p");
    title.textContent = method.name; text.textContent = method.description; article.append(title,text);
    if (method.url) {
      const link = document.createElement("a"); link.href = method.url; link.target = "_blank"; link.rel = "noopener noreferrer";
      link.textContent = method.paper; article.append(link);
    }
    list.append(article);
  }
  for (const button of document.querySelectorAll('[data-open-references], #method-references')) button.addEventListener("click", () => dialog.showModal());
  document.getElementById("references-close").addEventListener("click", () => dialog.close());
}
