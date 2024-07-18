from rdkit import Chem
import numpy as np
import rdkit


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    return None


def generate_cyclic_shifts(smiles):
    """Generate all cyclic shifts of a SMILES string."""
    smiles = smiles.replace('*', '')
    shifts = [smiles[i:] + smiles[:i] for i in range(len(smiles))]
    valid_shifts = [shift for shift in shifts if Chem.MolFromSmiles(shift) is not None]
    return valid_shifts


def generate_mirror_image(smiles):
    """Generate the mirror image of a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mirrored = Chem.MolToSmiles(Chem.Mol(mol, True), canonical=True)
        return mirrored
    return None


def canonicalize_all_variants(smiles):
    """Canonicalize all cyclic shifts and mirror image shifts of a SMILES string."""
    shifts = generate_cyclic_shifts(smiles)
    mirror_image = generate_mirror_image(smiles)
    if mirror_image:
        mirror_shifts = generate_cyclic_shifts(mirror_image)
    else:
        mirror_shifts = []
    all_variants = shifts + mirror_shifts
    canonical_variants = [canonicalize_smiles(variant) for variant in all_variants]
    return min(canonical_variants)  # Return the smallest canonical form


def smiles_match(smiles_1: str, smiles_2: str) -> bool:
    """
    True if SMILES represent the same material
    """
    canonical_1 = canonicalize_all_variants(smiles_1)
    canonical_2 = canonicalize_all_variants(smiles_2)
    return canonical_1 == canonical_2


def connect_mols(emol: rdkit.Chem.rdchem.EditableMol, endcap1: str, endcap2: str) -> rdkit.Chem.rdchem.Mol:
    combined_mol = emol.GetMol()
    ec1_idx = []
    ec2_idx = []

    for idx, atom in enumerate(combined_mol.GetAtoms()):
        if atom.GetSymbol() == endcap1:
            ec1_idx.append(idx)
        if atom.GetSymbol() == endcap2:
            ec2_idx.append(idx)

    atoms_to_remove = [ec1_idx[0], ec2_idx[1]]
    atoms_to_remove.sort(reverse=True)

    repl_idx = []
    for b in combined_mol.GetBonds():
        if b.GetEndAtomIdx() in atoms_to_remove:
            repl_idx.append(b.GetBeginAtomIdx())
        if b.GetBeginAtomIdx() in atoms_to_remove:
            repl_idx.append(b.GetEndAtomIdx())

    for atom in atoms_to_remove:
        emol.RemoveAtom(atom)

    corr_idx = [None] * 2
    for i, idx in enumerate(repl_idx):
        count = sum(1 for atom_num in atoms_to_remove if atom_num < idx)
        corr_idx[i] = count

    add_idx = np.array(repl_idx) - np.array(corr_idx)

    emol.AddBond(int(add_idx[0]), int(add_idx[1]), order=Chem.rdchem.BondType.SINGLE)

    connected_mol = emol.GetMol()
    return connected_mol


def count_main_chain_atoms(monomer_smiles: str) -> int:
    mol = Chem.MolFromSmiles(monomer_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {monomer_smiles}")

    au_index = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'Au')
    cu_index = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cu')

    shortest_path = Chem.rdmolops.GetShortestPath(mol, au_index, cu_index)
    main_chain_atoms = len(shortest_path)
    return main_chain_atoms


def rdkit_polymerize(start_mol: rdkit.Chem.rdchem.Mol, dp: int) -> rdkit.Chem.rdchem.Mol:
    polymerized_mol = start_mol
    mers = 0
    while mers < dp - 1:
        combo = Chem.CombineMols(polymerized_mol, start_mol)
        edcombo = Chem.EditableMol(combo)
        polymerized_mol = connect_mols(edcombo, 'Cu', 'Au')
        mers += 1
    return polymerized_mol


def required_polymerization(smiles):
    length_excluding_asterisks = len(smiles.replace('*', ''))
    if length_excluding_asterisks <= 5:
        return True, 10 // length_excluding_asterisks
    else:
        return False, 0


def create_polymer_smiles(monomer_smiles: str, degree_of_polymerization: int) -> str:
    monomer_smiles = monomer_smiles.replace('*', '[Au]', 1).replace('*', '[Cu]', 1)
    start_mol = Chem.MolFromSmiles(monomer_smiles)
    polymer_mol = rdkit_polymerize(start_mol, degree_of_polymerization)
    polymer_smiles = Chem.MolToSmiles(polymer_mol)
    polymer_smiles = polymer_smiles.replace('[Au]', '*')
    polymer_smiles = polymer_smiles.replace('[Cu]', '*')
    polymer_mol = Chem.MolFromSmiles(polymer_smiles)
    Chem.SanitizeMol(polymer_mol)
    polymer_smiles = Chem.MolToSmiles(polymer_mol)
    return polymer_smiles


# Process each SMILES string
canonical_smiles_list = []
canonical_without_ends = []
duplicated_smiles = []


for s in smiles_list: # smiles_list is the comprehensive list of generated
    # SMILES strings with the same order of generation in the first +
    # second iterations
    canonical_s = canonicalize_all_variants(s)
    canonical_s_without_ends = canonicalize_all_variants(s.replace('*', ''))

    if canonical_s in canonical_smiles_list or canonical_s_without_ends in canonical_without_ends:
        duplicated_smiles.append(s)
    else:
        canonical_smiles_list.append(canonical_s)
        canonical_without_ends.append(canonical_s_without_ends)

        if required_polymerization(s)[0]:
            for n in range(2, required_polymerization(s)[1]):
                polymer_smiles = create_polymer_smiles(monomer_smiles=s, degree_of_polymerization=n)
                canonical_polymer_smiles = canonicalize_all_variants(polymer_smiles)
                canonical_polymer_smiles_without_ends = canonicalize_all_variants(polymer_smiles.replace('*', ''))

                if canonical_polymer_smiles not in canonical_smiles_list:
                    canonical_smiles_list.append(canonical_polymer_smiles)
                    canonical_without_ends.append(canonical_polymer_smiles_without_ends)


print("\nDuplicated SMILES:")
for smiles in duplicated_smiles:
    print(smiles)
