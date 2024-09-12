import enum

import ase
import ase.optimize
import numpy as np
import rdkit2ase
import tqdm
from ase import units
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.langevin import Langevin
from mace.calculators import mace_mp
from pydantic import Field
from zndraw import Extension, ZnDraw


def freeze_copy_atoms(ref: ase.Atoms) -> ase.Atoms:
    """Create a copy of the atoms object."""

    atoms = ase.Atoms(
        ref.get_atomic_numbers(),
        ref.get_positions(),
        pbc=ref.get_pbc(),
        cell=ref.get_cell(),
    )
    if ref.calc is not None:
        results = {}
        if "energy" in ref.calc.results:
            results["energy"] = ref.calc.results["energy"]
        if "forces" in ref.calc.results:
            results["forces"] = ref.calc.results["forces"]
        atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


class Optimizer(str, enum.Enum):
    """Available optimizers for geometry optimization."""

    LBFGS = "LBFGS"
    FIRE = "FIRE"
    BFGS = "BFGS"


class Models(str, enum.Enum):
    """Available models for energy and force prediction."""

    MACE_MP_0 = "MACE-MP-0"
    LJ = "LJ"


class MolecularDynamics(Extension):
    """Run molecular dynamics in the NVT ensemble using Langevin dynamics."""

    model: Models = Models.MACE_MP_0
    temperature: float = 300
    time_step: float = 0.5
    n_steps: int = 100
    friction: float = 0.002

    upload_interval: int = 10

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
        atoms_cache = []

        for _ in tqdm.trange(self.n_steps):
            dyn.run(1)
            atoms_cache.append(freeze_copy_atoms(atoms))
            if len(atoms_cache) == self.upload_interval:
                vis.extend(atoms_cache)
                atoms_cache = []
        vis.extend(atoms_cache)


class GeomOpt(Extension):
    """Run geometry optimization"""

    model: Models = Models.MACE_MP_0

    optimizer: Optimizer = Optimizer.LBFGS
    fmax: float = 0.05

    upload_interval: int = 10

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

        atoms_cache = []

        for idx, _ in enumerate(dyn.irun(fmax=self.fmax)):
            atoms_cache.append(freeze_copy_atoms(atoms))
            if len(atoms_cache) == self.upload_interval:
                vis.extend(atoms_cache)
                atoms_cache = []
            if idx > 100:  # max 100 steps
                break
        vis.extend(atoms_cache)


class AddFromSMILES(Extension):
    """Place a molecule from a SMILES at all points."""

    SMILES: str = Field(..., description="SMILES string of the molecule to add")

    def run(self, vis: "ZnDraw", **kwargs) -> None:
        molecule = rdkit2ase.smiles2atoms(self.SMILES)

        scene = vis.atoms

        points = vis.points
        if len(points) == 0:
            points = [np.array([0, 0, 0])]

        for point in points:
            molecule_copy = molecule.copy()
            molecule_copy.translate(point)
            scene.extend(molecule_copy)

        if hasattr(scene, "connectivity"):
            del scene.connectivity

        vis.append(freeze_copy_atoms(scene))
        vis.bookmarks = vis.bookmarks | {vis.step: "AddFromSMILES"}


class Solvate(Extension):
    """Solvate the current scene."""

    solvent: str = Field(..., description="Solvent to use (SMILES)")
    count: int = Field(
        10, ge=1, le=500, description="Number of solvent molecules to add"
    )
    density: float = Field(789, description="Density of the solvent")
    pbc: bool = Field(True, description="Whether to use periodic boundary conditions")
    tolerance: float = Field(2.0, description="Tolerance for the solvent")

    def run(self, vis: "ZnDraw", **kwargs) -> None:
        scene = vis.atoms
        solvent = rdkit2ase.smiles2atoms(self.solvent)

        scene = rdkit2ase.pack(
            [[scene], [solvent]],
            [int(len(scene) > 0), self.count],
            density=self.density,
            pbc=self.pbc,
            tolerance=self.tolerance,
        )

        vis.append(scene)
        vis.bookmarks = vis.bookmarks | {vis.step: "Solvate"}


if __name__ == "__main__":
    import os

    print("Downloading the MACE-MP-0 model")
    mace_mp_0_model_calc = mace_mp(
        model="medium", dispersion=False, default_dtype="float32", device="cuda"
    )
    print("Model downloaded")

    url = os.environ.get("ZNDRAW_URL")
    token = os.environ.get("ZNDRAW_TOKEN")
    auth_token = os.environ.get("ZNDRAW_AUTH_TOKEN")

    vis = ZnDraw(
        url=url,
        token=token,
        auth_token=auth_token,
    )

    print(f"Registering {MolecularDynamics}")
    vis.register_modifier(
        MolecularDynamics,
        run_kwargs={"calc": mace_mp_0_model_calc},
        public=True,
    )
    vis.socket.sleep(5)

    print(f"Registering {AddFromSMILES}")
    vis.register_modifier(AddFromSMILES, public=True)
    vis.socket.sleep(5)

    print(f"Registering {Solvate}")
    vis.register_modifier(Solvate, public=True)
    vis.socket.sleep(5)

    print(f"Registering {GeomOpt}")
    vis.register_modifier(
        GeomOpt,
        run_kwargs={"calc": mace_mp_0_model_calc},
        public=True,
    )

    print("All modifiers registered. Starting the main loop...")
    vis.socket.wait()
