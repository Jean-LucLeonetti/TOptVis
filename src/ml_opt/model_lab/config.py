from dataclasses import dataclass
import math


CATALOG = [
    ("pod", "POD Galerkin", "POD", "A frozen displacement basis with exact reduced element assembly."),
    ("energy_pod", "Energy-weighted POD", "POD", "Selects a basis using the full-solid stiffness energy inner product."),
    ("local_pod", "Local POD", "POD", "Clusters material layouts and fits a separate Galerkin basis in each cluster."),
    ("adaptive_pod", "Adaptive POD", "POD", "Starts from POD; uses FOM when the residual is high and enriches at accepted optimisation iterates."),
    ("pod_mlp", "POD + MLP", "Neural network", "A small network predicts coefficients in a fixed POD displacement basis."),
    ("cnn", "CNN", "Neural network", "A convolutional network maps material and load fields to nodal displacement."),
    ("unet", "U-Net", "Neural network", "An encoder-decoder with spatial skip connections predicts displacement."),
    ("fno", "Fourier neural operator", "Neural operator", "Spectral convolution blocks learn the material-to-displacement mapping."),
    ("deeponet", "DeepONet", "Neural operator", "A field encoder and coordinate network learn a separable displacement operator."),
    ("transformer", "Mesh transformer", "Transformer", "A compact pooled-token transformer attends from mesh nodes to learned field tokens."),
]
POD_FAMILIES = {"pod", "energy_pod", "local_pod", "adaptive_pod"}
FAMILIES = {item[0] for item in CATALOG}
SIGNATURE = ("nx", "ny", "length", "height", "young_modulus", "minimum_modulus", "poisson_ratio", "thickness")


@dataclass(frozen=True)
class TrainOptions:
    family: str = "fno"
    epochs: int = 80
    batch_size: int = 8
    learning_rate: float = .001
    width: int = 24
    latent: int = 24
    fourier_modes: int = 6
    layers: int = 3
    energy_target: float = .99999
    max_modes: int = 24
    clusters: int = 3
    validation_fraction: float = .25
    seed: int = 7
    energy_loss_weight: float = .1
    residual_loss_weight: float = 0.
    patience: int = 15
    device: str = "cpu"
    residual_threshold: float = .05
    adaptive_max_modes: int = 64

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError("Unknown model family")
        for name,low,high in (("epochs",1,2000),("batch_size",1,32),("width",8,64),
            ("latent",4,128),("fourier_modes",1,16),("layers",1,6),("max_modes",1,128),
            ("clusters",1,8),("seed",0,2**31-1),("patience",1,2000),("adaptive_max_modes",1,128)):
            value=getattr(self,name)
            if type(value) is not int or not low<=value<=high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")
        for name,low,high,closed in (("learning_rate",0,.1,False),("energy_target",0,1,False),
            ("validation_fraction",0,.5,False),("energy_loss_weight",0,100,True),
            ("residual_loss_weight",0,100,True),("residual_threshold",0,1,False)):
            value=getattr(self,name)
            if not math.isfinite(value) or value>high or (value<low if closed else value<=low):
                raise ValueError(f"Invalid {name}")
        if self.device not in ("cpu","cuda","auto"):
            raise ValueError("Device must be cpu, cuda or auto")
        if self.family=="transformer" and self.width%4:
            raise ValueError("Transformer width must be divisible by four")
        if self.family=="adaptive_pod" and self.adaptive_max_modes<self.max_modes:
            raise ValueError("Adaptive mode limit must be at least the initial mode limit")
