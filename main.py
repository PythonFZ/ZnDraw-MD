import mace_models
from ase.md.langevin import Langevin
from zndraw import ZnDraw
from zndraw.modify import UpdateScene
from ase import units
import ase
import typing as t
import tqdm
import ase.optimize
import enum
from ase.calculators.lj import LennardJones


class Optimizer(enum.Enum):
    """Available optimizers for geometry optimization."""

    LBFGS = "LBFGS"
    FIRE = "FIRE"
    BFGS = "BFGS"


class Models(enum.Enum):
    """Available models for energy and force prediction."""

    MACE_MP_0 = "MACE-MP-0"
    LJ = "LJ"


class MolecularDynamics(UpdateScene):
    """Run molecular dynamics in the NVT ensemble using Langevin dynamics."""

    discriminator: t.Literal["MolecularDynamics"] = "MolecularDynamics"
    model: Models = Models.MACE_MP_0
    temperature: float = 300
    time_step: float = 0.5
    n_steps: int = 100
    friction: float = 0.002

    def run(self, vis: ZnDraw, calc, **kwargs):
        if self.model.value == "LJ":
            calc = LennardJones()
        if self.n_steps > 1000:
            raise ValueError("n_steps should be less than 1000")
        if len(vis) > vis.step + 1:
            del vis[vis.step + 1 :]
        atoms = vis.atoms
        if len(atoms) > 1000:
            raise ValueError("Number of atoms should be less than 1000")
        atoms.calc = calc
        dyn = Langevin(
            atoms,
            timestep=self.time_step * units.fs,
            temperature_K=self.temperature,
            friction=self.friction,
        )
        vis.bookmarks = vis.bookmarks | {vis.step: "Molecular dynamics"}
        for _ in tqdm.trange(self.n_steps):
            dyn.run(1)
            vis.append(
                ase.Atoms(
                    atoms.get_atomic_numbers(),
                    atoms.get_positions(),
                    pbc=atoms.get_pbc(),
                    cell=atoms.get_cell(),
                )
            )


class GeomOpt(UpdateScene):
    """Run geometry optimization"""

    discriminator: t.Literal["GeomOpt"] = "GeomOpt"
    model: Models = Models.MACE_MP_0

    optimizer: Optimizer = Optimizer.LBFGS
    fmax: float = 0.05

    def run(self, vis: ZnDraw, calc, **kwargs) -> None:
        if self.model.value == "LJ":
            calc = LennardJones()
        optimizer = getattr(ase.optimize, self.optimizer.value)
        atoms = vis.atoms
        if len(atoms) > 1000:
            raise ValueError("Number of atoms should be less than 1000")
        atoms.calc = calc
        dyn = optimizer(atoms)
        vis.bookmarks = vis.bookmarks | {vis.step: "Geometric optimization"}

        for idx, _ in enumerate(dyn.irun(fmax=self.fmax)):
            vis.append(
                ase.Atoms(
                    atoms.get_atomic_numbers(),
                    atoms.get_positions(),
                    pbc=atoms.get_pbc(),
                    cell=atoms.get_cell(),
                )
            )
            if idx > 100:  # max 100 steps
                break


if __name__ == "__main__":
    mace_mp_0_model_calc = mace_models.LoadModel.from_rev(
        "MACE-MP-0", remote="https://github.com/RokasEl/MACE-Models.git"
    ).get_calculator()
    vis = ZnDraw(
        url="https://zndraw.icp.uni-stuttgart.de",
    )
    vis.register_modifier(
        MolecularDynamics,
        run_kwargs={"calc": mace_mp_0_model_calc},
        default=True,  # type: ignore
    )
    vis.socket.sleep(5)
    vis.register_modifier(
        GeomOpt,
        run_kwargs={"calc": mace_mp_0_model_calc},
        default=True,  # type: ignore
    )
    print("All modifiers registered. Starting the main loop...")
    while True:
        try:
            vis.socket.emit("modifier:available", vis._available)
        except Exception as e:
            print(32 * "-")
            print("Not connected to ZnDraw: %s", e)
            print("Trying to reconnect...")
            vis.reconnect()
            print("Reconnected to ZnDraw")
        finally:
            vis.socket.sleep(10)
