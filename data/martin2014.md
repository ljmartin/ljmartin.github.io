# Molecular dynamics simulation of the partitioning of benzocaine and phenytoin into a lipid bilayer

**Lewis J. Martin, Rebecca Chao, Ben Corry***

Research School of Biology, Australian National University, Canberra 0200, Australia

*Corresponding author. E-mail address: ben.corry@anu.edu.au

---

## Highlights

- The local anaesthetics benzocaine and phenytoin partition into a lipid membrane.
- Both drugs face a barrier to cross the bilayer centre.
- Both drugs can pull water into the membrane and create extended water chains.
- Once in the membrane they can alter bilayer properties or reach target proteins.
- Results depend on atomic charges showing importance of validating new drug parameters.

---

## Graphical Abstract

> **Graphical Abstract Figure**
> 
> Interpretation: A molecular dynamics simulation snapshot showing a cross-section of a lipid bilayer with embedded drug molecules. The bilayer structure shows lipid headgroups (colored spheres) at the top and bottom with a hydrophobic core in the middle. Drug molecules are shown positioned within the bilayer, with colors indicating atom types. The visualization demonstrates how local anaesthetic drugs partition into and interact with the lipid membrane environment.
> 
> Caption: Graphical representation of the simulation system showing drug molecules partitioning into a lipid bilayer membrane.

---

## Abstract

Molecular dynamics simulations were used to examine the partitioning behaviour of the local anaesthetic benzocaine and the anti-epileptic phenytoin into lipid bilayers, a factor that is critical to their mode of action. Free energy methods are used to quantify the thermodynamics of drug movement between water and octanol as well as for permeation across a POPC membrane. Both drugs are shown to favourably partition into the lipid bilayer from water and are likely to accumulate just inside the lipid headgroups where they may alter bilayer properties or interact with target proteins. Phenytoin experiences a large barrier to cross the centre of the bilayer due to less favourable energetic interactions in this less dense region of the bilayer. Remarkably, in our simulations both drugs are able to pull water into the bilayer, creating water chains that extend back to bulk, and which may modify the local bilayer properties. We find that the choice of atomic partial charges can have a significant impact on the quantitative results, meaning that careful validation of parameters for new drugs, such as performed here, should be performed prior to their use in biomolecular simulations.

---

## 1. Introduction

Local anaesthetic, anti-epileptic and anti-arrhythmic drugs are known to target voltage-gated sodium channels residing in cell membranes [1]. The potency of these compounds is positively correlated with lipophilicity, [2–5] indicating that their ability to enter or cross membranes is critical to their mechanism of action. It is likely that the efficacy of these compounds is directly related to their ability to reach their protein target, where they act to block currents or alter channel kinetics [6]. Electrophysiological studies have shown that these compounds can reach their binding site in the centre of the sodium channel pore by one of two routes [7]. The first involves passing through the intracellular activation gate from the cytoplasm when the channel is open. The second involves passing through an alternative hydrophobic access route directly from the lipid [7] to yield tonic block of resting channels. The hydrophobic pathway through the protein was given clarity in recent studies of bacterial channels, [8–11] and the ability of a drug to find this passage is critically dependent upon its apportionment into the lipid bilayer as well as its ability to fit through the gap in the protein. In each case the compounds have to be able to enter or cross the membrane and so understanding the partitioning of these compounds between water and lipid is of great importance to understanding their mode of action.

> **Figure 1**
> 
> Interpretation: Panel A shows the molecular structure of benzocaine rendered as a 3D ball-and-stick model with cyan carbons, red oxygens, blue nitrogen, and white hydrogens. Panel B shows the more complex molecular structure of phenytoin, which contains two fused aromatic rings with cyan carbons, red oxygens, blue nitrogens, and white hydrogens. Panel C displays a cross-sectional view of the simulation system showing a lipid bilayer with the embedded drug molecule colored by atom type (carbon cyan, oxygen red, nitrogen blue, hydrogen white). Na+ ions are shown in yellow, Cl- in cyan, phosphorus in brown, and lipid carbons in grey. A transparent surface indicates the volume sampled by water molecules around the membrane system.
> 
> Caption: Images of (A) benzocaine, (B) phenytoin and (C) the simulation system. The drugs are coloured according to atom type (carbon cyan, oxygen red, nitrogen blue and hydrogen white). In (C) Na+ is shown in yellow, Cl− in cyan, phosphorus in brown and the lipid carbons in grey. The transparent surface indicates the volume sampled by water.

In addition to direct interaction between local anaesthetics and sodium channels, the cumulative effect of drugs on the lipid bilayer is also implicated in anaesthetic action [12]. Even before the nature of cell membranes was understood, a correlation between the hydrophobicity of an anaesthetic agent and its potency was discovered [13,14]. More recently anaesthetic action has been tied to changes in the lateral pressure of a lipid bilayer, a property that can vary along the membrane normal and possibly affect channel activity by altering the conformational landscape [15–17]. The partitioning of anaesthetics into membranes could also alter other local properties of the bilayer and thus modify the behaviour of a range of membrane bound proteins.

As the ability of a huge number of drugs to passively cross the bilayer is essential to their activity, a great many simulation and experimental investigations of lipid permeability and water/lipid partitioning have been conducted (see [18] and references therein). More specifically a number of simulation studies have examined the interaction between sodium channel targeting drugs and membranes at atomic detail. These have included investigations into the thermodynamics of insertion of benzocaine into DPPC and mixed DPPC/DPPS lipid bilayers, [19,20] and the likely positioning and influence of a DMPC bilayer of charged and uncharged lidocaine [21,22] and articaine [23] relative to a DMPC bilayer. Indeed, the free energy of solute transfer from the water phase into the membrane has been calculated for various anaesthetic compounds and it was found that the work to create a cavity able to locate a permeant solute is lower inside the membrane than in water, whereas the electrostatic contribution to the solute transfer increases monotonically going from water into the membrane interior. A balance between these two opposite effects causes dipolar compounds to accumulate at the water/membrane interface, whereas apolar compounds resided predominantly in the membrane core [18]. This behaviour was qualitatively related to the anaesthetic power of these compounds, with the most polar compounds that concentrate at the interface being the most powerful [24]. Polar anaesthetics thus tend to experience a barrier to cross the centre of the bilayer due either to the removal of interactions with the polar components of the bilayer or the compound reducing the lipid mobility when placed in the membrane core. There appear to be specific relationships between solute size on mobility and partitioning, although there is still some doubt as to which property is most strongly altered by solute size [18].

The choice of force field parameters can have a large influence on the results of molecular dynamics simulations, but this aspect has not been thoroughly explored in the context of local anaesthetic partitioning. The effect of properties such as partial atomic charge on the spontaneity of lipid partition can be tested in simulation and measured against values determined by experiment to determine their accuracy. The specific studies of local anaesthetics described above have employed united atom force fields and it is possible that this can alter the effective polarity and size of the compounds. In addition, these parameters were not verified for their ability to reproduce measured partition coefficients.

Here we examine the behaviour of the local anaesthetic benzocaine (Fig. 1A) and the anti-epileptic phenytoin (Fig. 1B) in a POPC membrane and calculate the free energies of drug partitioning and bilayer permeation. The polar surface areas of benzocaine and phenytoin are 54.3 Å² (24.0% of total) and 69.5 Å² (23.4% of total) respectively, meaning that benzocaine would be expected to cross membranes more easily. We calculate the results using a range of atomic partial charges to examine how the choice of these parameters alters the thermodynamics of partitioning. The comparison between linear benzocaine and the bulkier phenytoin also allows us to examine the influence of solute size. Both drugs are shown to favourably partition into the lipid bilayer from water and are likely to accumulate just inside the lipid headgroups. From this position they may alter the bilayer properties or directly enter voltage-gated sodium channels through the hydrophobic access route. Phenytoin experiences a large barrier to cross the centre of the bilayer due to less favourable energetic interactions in this less dense region of the bilayer. Remarkably, in our simulations both drugs are able to pull water into the bilayer, creating water chains that extend back to bulk, and which may modify the local bilayer properties.

---

## 2. Method

### 2.1. Partition coefficients

Water/octanol partition coefficients for benzocaine and phenytoin were determined by calculation of the difference in the free energy of solvation in octanol and in water. Each of these values was collected by free energy perturbation experiments in separate simulations representing the water and octanol phases.

The free energy of transfer from water to octanol can be calculated according to:

$$\Delta G = \Delta G_{wat} - \Delta G_{oct}$$ (1)

where ΔG_wat is the free energy of hydration and ΔG_oct is the free energy of solvation in octanol. The resulting net free energy values can be combined into a partition coefficient, P_ow, using the following relation:

$$\text{Log}(P_{ow}) = \frac{\Delta G}{2.303RT}$$ (2)

where R is the universal gas constant and T is the simulation temperature.

The coordinates for the alcohol phase systems were generated using packmol [25] with a box size of 48 × 48 × 48 Å, using 463 octanol molecules placed around the drug and separated by 2 Å. These systems were equilibrated for 20 ns, after which the system pressure and total number of hydrogen bonds had stabilised. The coordinates for the aqueous phase systems were generated using the Solvate plugin in VMD [26] with a box size of 48 × 48 × 48 Å using the TIP3P water model. The aqueous phases were ionized using the Autoionize plugin in VMD, to a physiological salt concentration of 0.15 M. During a 5 ns equilibration, the density and number of hydrogen bonds in these systems stabilised after 2–3 ns.

The free energy of solvation for both drugs in water and octanol was determined using free energy perturbation [27]. For each set of atomic charges free energy perturbation was performed three times in both the forward and reverse directions. The reaction was stratified into 40 windows, with lambda values advancing by 0.025 in each. A softcore potential was utilised to avoid end-point catastrophes. This scaled down the electrostatic interactions of annihilating particles from lambda = 0.5 to lambda = 1, and the van der Waals interactions from lambda = 0 to lambda = 1. Each window was equilibrated for 0.5 ns, and data collection/ensemble averaging took place for a further 1 ns per window.

### 2.2. Drug parameters

Partial charges for benzocaine and phenytoin were determined in 4 ways, yielding four separate parameter sets varying in polarity. In the first of these, atomic charges were determined by analogy to similar atom types in the CHARMM general forcefield (CGFF) [28]. The remaining three sets of partial charges were all obtained by conducting ab-initio geometry optimisation and then fitting the charges to the electrostatic potential using the program GAUSSIAN03 [29] and the Merz–Kollman (MK) electrostatic fitting method [30,31]. The difference between these charge sets arises from the theory and basis set used in the calculations which were either B3LYP/6-31+G* (referred to as B3LYP from here), HF/6-31G, and HF/6-31+G*. In the last case atomic charges were averaged across atoms with the same chemical environment. The values of the charges used are given in Tables S1 & S2. A consistent order of polarity is seen amongst these charge sets for benzocaine and phenytoin, with the CGFF charges having the lowest magnitude, B3LYP having intermediate magnitude and the HF methods yielding the most polar molecules. van der Waals and Bond parameters were taken from similar atom types in the CHARMM27 force field [32].

### 2.3. Lipid partitioning

The free energy (potential of mean force) for drug partitioning into a lipid bilayer for each drug was determined using the method of umbrella sampling, [33] with a separate set of simulations for each drug. In these, the centre of mass of the drug was held at 1 Å intervals in the direction of the membrane normal, as calculated from the distance between the centre of mass of the drug and the lipid phosphorus atoms, using a harmonic potential with a force constant of 2 kcal mol⁻¹ Å⁻². The drug was equilibrated at each position for 10 ns prior to 30 ns of data collection. Collective analysis of the drug positions was made with the weighted histogram analysis method [34,35] using the implementation of Grossfield [36] to yield the final free energy profiles. Error bars were calculated by dividing the 30 ns trajectory into 10 ns blocks and determining the standard error in the free energy found from each of these. All results are calculated from bulk into the centre of the bilayer, and mirrored to produce a PMF traversing the entire bilayer. The 53 × 53 Å POPC bilayer was created using the membrane builder plugin in VMD and solvated with TIP3P water and 150 mM NaCl to yield a final system that was 53 × 53 × 95 Å in size after equilibration as depicted in Fig. 1C.

All simulations were run with NAMD 2.8 [37], using the CHARMM27 forcefield for octanol [32], CHARMM36 for lipids [38], ion parameters from Joung and Cheatham [39] and with water described by the TIP3P model. Periodic boundary conditions were applied, with the particle mesh Ewald scheme for calculating electrostatic interactions. Constant temperature (298 K) and pressure (1 atm) were maintained (NPT ensemble) using Langevin dynamics and a Langevin piston. Bonds to hydrogen atoms were fixed to allow for a timestep of 2 fs. The van der Waals cutoff was set at 12 Å. The polar surface area of each molecule was calculated using the SurfArea program from Cidtrux Pharminformatics.

> **Table 1**
> 
> The effect of atomic charges on the octanol–water partition coefficient of phenytoin and benzocaine.

| Charge determination | Phenytoin |  | Benzocaine |  |
|---------------------|-----------|-------------|------------|-------------|
| | ΔG_oct − ΔG_wat (kcal/mol) | Log(P_ow) | ΔG_oct − ΔG_wat (kcal/mol) | Log(P_ow) |
| HF 6-31G | 3.58 ± 0.36 | 2.62 ± 0.26 | 1.66 ± 0.13 | 1.22 ± 0.09 |
| HF 6-31+G* | 3.28 ± 0.41 | 2.41 ± 0.30 | 3.49 ± 0.29 | 2.56 ± 0.17 |
| B3LYP | 5.49 ± 0.35 | 4.03 ± 0.26 | 4.52 ± 0.15 | 3.31 ± 0.11 |
| CGFF | 5.82 ± 0.12 | 4.27 ± 0.08 | 3.02 ± 0.14 | 2.21 ± 0.10 |
| Experiment |  | 1.92–2.47 |  | 1.86–1.95 |

---

## 3. Results

### 3.1. Water/octanol partitioning

The effect of different atomic charges on the octanol–water partition coefficient of benzocaine and phenytoin is demonstrated in Table 1. In each case the forward and reverse reactions agreed to within 1–2 kcal/mol (Tables S3 & S4). As expected, more polar drug molecules – those with atomic charges of greater magnitude – tended to have greater interaction energies with both water and octanol. This is reflected in the radial distribution functions measuring hydrogen bonding to the octanol hydroxyl moiety, which show greater peaks in the Hartree–Fock-paramaterised charge sets than in the less-polar B3LYP or CGFF charge sets (Figure S3). The difference between these interaction energies led to the calculated changes in Log(P_ow) with the less polar versions of the drugs showing stronger partitioning into octanol than the more polar ones.

Observed Log(P_ow) values for phenytoin fall between 1.92 and 2.47 [40–42]. Benzocaine has a comparatively lower affinity for the organic phase, with observed log(P_ow) values of between 1.86 and 1.95 [43–45]. In our simulation system we get the closest agreement with experiment with the HF charge set for phenytoin and the CGFF charge set for benzocaine. To better understand how the choice of charges might influence lipid partitioning we compare the CGFF and HF charge sets in our subsequent partitioning studies.

> **Figure 2**
> 
> Interpretation: Panel A displays atomic density profiles as a function of position along the bilayer normal. Four curves are shown: water (blue) with high density outside the bilayer and zero in the hydrophobic core; phosphorus (black) representing the phosphate groups in the headgroup region; CH2 groups (green) showing the lipid tails with a minimum at the center; and oxygen (red) showing the distribution of oxygen atoms. The x-axis shows position in Angstroms from -30 to +30, with 0 representing the bilayer center. Panel B shows free energy profiles with error bars for benzocaine and phenytoin using different charge parameter sets (CGFF and HF). All curves show minima around ±15 Å (near the headgroup region) and maxima at 0 Å (bilayer center), indicating an energy barrier at the center. The different charge sets show varying depths of the energy minima, with CGFF charges producing deeper minima than HF charges.
> 
> Caption: Atomic densities (A) and free energy profiles for benzocaine and phenytoin as a function of position relative to the bilayer centre (B). Results for two different sets of partial charges for each drug are shown and error bars are calculated from the standard error derived from breaking the trajectories into 10 ns segments.

### 3.2. Partitioning into lipid

To determine if benzocaine and phenytoin will partition into a POPC membrane we determined the free energy of each drug as a function of its position relative to the bilayer. As can be seen by the position of the free energy minimum in Fig. 2 both drugs prefer to reside inside the bilayer rather than in water. The most favoured position to reside is just inside the lipid headgroups, as shown pictorially in Fig. 3 where the hydrophobic portion of the molecule can be buried in the interior of the membrane while the polar portion of the drug can interact with the lipid headgroups and water. Compared to bulk, this location is favoured by 2–4 kcal/mol for benzocaine and 3–6 kcal/mol for phenytoin depending on the specific choice of atomic charges. Beyond this point both drugs see an energy barrier to pass through the centre of the bilayer before finding a second minimum in the other bilayer leaflet. The depth of the free energy minima seen with the different atomic charges relate directly to the magnitude of the atomic partial charges. The less polar CGFF charges yield a deeper energy minimum, implying stronger partitioning into the bilayer, while the more highly charged HF charges yield less deep minima as expected.

> **Figure 3**
> 
> Interpretation: Two molecular dynamics simulation snapshots showing the location of drug molecules within the lipid bilayer. Panel A shows benzocaine positioned at the free energy minimum, with the drug molecule (shown as cyan/red/blue vdW spheres) located near the lipid headgroups. Phosphorus atoms are shown in brown, remaining lipid atoms as grey lines, and water as red and white lines. Panel B shows phenytoin in a similar position within the bilayer. Both images demonstrate how the drugs sit at the interface between the aqueous phase and the hydrophobic membrane core, with the polar portions of the drugs interacting with the lipid headgroups and water while the hydrophobic portions are buried in the membrane interior.
> 
> Caption: Simulation snapshots showing (A) benzocaine and (B) phenytoin at the position of the free energy minimum. In each case the drug is shown as coloured vdW spheres, the phosphorus atoms in brown. The remaining lipid atoms are shown as grey lines and water as red and white lines.

Not surprisingly, the desire of both drugs to have their polar end interact with either the lipid headgroups or water means that the drug has preferred orientations relative to the bilayer that change as a function of position. This is pictured in Fig. 4 in which the average orientation, standard deviation as well as the maximum and minimum values of the drug relative to the plane of the bilayer are shown. When in bulk the drugs have no preferred orientation, however this changes near the position of the free energy minima where the polar/apolar asymmetry in each drug aligns the molecule to point along the bilayer normal.

> **Figure 4**
> 
> Interpretation: Four scatter plots showing the orientation of each drug relative to the plane of the membrane as a function of position. Each panel represents a different drug and charge set combination: Panel A (Benzocaine CGFF), Panel B (Benzocaine HF), Panel C (Phenytoin CGFF), and Panel D (Phenytoin HF). The y-axis shows orientation in degrees (0-180°), and the x-axis shows position in Ångstroms (-30 to +30). The main data points show the mean orientation values with error bars representing standard deviation. The grey shaded regions in the background indicate the range of values sampled during the 30 ns simulations (minimum to maximum). All panels show that when drugs are in bulk water (far left and right), orientations are random (scattered around 90°), but when positioned near the headgroup region (around ±15 Å), the drugs adopt specific orientations aligned with the bilayer normal (near 0° or 180°).
> 
> Caption: Orientations of each drug relative to the plane of the membrane as a function of its position. The orientation is defined by the line joining the most remote nitrogen atom to either the carbon on the far side on the aromatic ring (benzocaine) or the carbon atom joining the phenyl rings (phenytoin). The average value and data points show the mean value and standard deviation, while the grey regions show the range of values sampled during the 30 ns simulations.

One of the most surprising aspects of this study is that both benzocaine and phenytoin are able to coordinate with water molecules, even when in the centre of the bilayer. To demonstrate this we calculate the number of coordinating water molecules (defined as those within 2.8 Å of the drug) as a function of drug position as plotted in Fig. 5. As the drugs move through the membrane the number of coordinating water molecules decreases to the bilayer centre. However, even when buried deep in the membrane core, some water molecules are present around each drug. This is most obvious for the more polar version of phenytoin, where the average number of coordinating water molecules never drops below two. Examination of the simulation trajectories shows that in this case two water molecules remain hydrogen bonded to phenytoin (Fig. 6C) and move through the bilayer with the drug. Phenytoin has a slightly lower coordination number in the centre of the bilayer with the CGFF charges than with the HF charges, indicating that the less polar version is less likely to pull water through the bilayer. The average number of water molecules surrounding benzocaine in the centre of the membrane is less than for phenytoin and close to zero, but the standard deviation in the coordination number indicates that sometimes benzocaine also directly contacts water when deep in the hydrophobic membrane core.

> **Figure 5**
> 
> Interpretation: Three line graphs showing water coordination statistics. Panel A shows the number of water molecules coordinating benzocaine as a function of position, comparing CGFF (black squares) and HF (blue circles) charge sets. Panel B shows the same for phenytoin with CGFF (black squares) and HF (red circles) charges. Both panels show that water coordination is high in bulk water (around 15-18 molecules), decreases as the drug enters the membrane, and reaches a minimum near the bilayer center (0 Å) before increasing again in the opposite bulk water region. Error bars represent statistical uncertainty. Panel C shows the probability of a continuous water chain extending from the drug to bulk water as a function of position (0-14 Å from the bilayer center). Four curves are shown representing both drugs with both charge sets. All curves show that the probability increases with distance from the center, approaching 1.0 (certainty) as the drug nears the bulk water. Fitted curves (solid lines) are shown for each dataset.
> 
> Caption: The number of water molecules directly coordinating (A) benzocaine and (B) phenytoin as a function of drug position. (C) The proportion of the time that a continuous water chain extends from the drug to bulk as a function of drug position.

An even more remarkable finding is that not only can both drugs interact with water in the centre of the bilayer, but that some of the time this water forms a continuous chain that connects back to bulk. First evidence for this comes from the fact that the water contacting the drugs can exchange with water in bulk, even when the drug is in the bilayer core. These water chains are also seen visually in the simulation trajectories as shown in the snapshots in Fig. 6B & D. Here it can be seen that the bilayer distorts slightly, allowing water molecules to penetrate to the buried drug molecule. We note that when the drugs are in the centre of the membrane these water chains are transitory, forming and breaking a number of times during the simulation. To quantify this in Fig. 5C we plot the proportion of the time that a continuous water chain is formed from the drug to bulk as a function of drug position. For both drugs and both sets of atomic charges, this continuous chain forms the majority of the time until the drug is very close to the membrane centre. The probability of this chain forming drops dramatically near the bilayer centre, but never reaches zero. Presumably the chance of such chain forming will also be dependent upon the bilayer thickness. Such 'snorkelling' has been reported for charged compounds in experiment [46] and simulation [47]. This behaviour is also observed for charged and neutral amino acid residues on the surface of a protein [48,49]. To our knowledge, no examples of snorkelling by uncharged molecular solutes have been reported.

> **Figure 6**
> 
> Interpretation: Four molecular dynamics simulation snapshots arranged in a 2x2 grid showing drug molecules at the center of the bilayer. Panels A and B show benzocaine, while panels C and D show phenytoin. In panels A and C, snapshots are captured when there is no continuous water chain connecting the drug to bulk water—phenytoin (C) maintains contact with two water molecules (shown as red/white spheres) while benzocaine (A) has no coordinating waters. In panels B and D, snapshots show when a continuous water chain exists—these water molecules close to the drug are highlighted as larger red and white spheres forming a connected pathway from the drug through the membrane interior to the bulk water phase. In all panels, lipid phosphorus atoms are shown as brown spheres, lipid tails as grey lines, and water as small red and white lines.
> 
> Caption: Simulation snapshots showing (A & B) benzocaine and (C & D) phenytoin at the centre of the bilayer. In (A) and (C) snapshots are taken when there is no water chain to bulk and phenytoin is in contact with two water molecules and benzocaine none. In (B) and (D) snapshots are taken when a continuous water chain exists and water molecules close to the drug are highlighted.

As with previous studies in DPPC, benzocaine experiences a slight energy barrier when passing through the lipid headgroups [19,20]. However, the magnitude and shape of the profiles seen here are quite different to those in previous studies. The depth of the well seen for benzocaine here (2–4 kcal/mol) is less than in the previous study for DPPC (5.8 kcal/mol), and notably we see an energy barrier in the centre of the bilayer not present in the previous work. Much of this difference is likely due to the different force fields and parameters used in the studies. Of particular note, the previous work uses a united atom force field in which hydrogen atoms are merged with heavy atoms. This yields a less polar version of benzocaine which is likely to move further into the core of the bilayer, and the snorkelling phenomenon seen here is not reported in those studies.

To understand the reason for the barrier seen by both drugs in the centre of the bilayer, in Fig. 7 we plot the total interaction energy of the drug with the rest of the system as well as the specific drug–water and drug–lipid interactions. While these interaction energies are only one factor contributing to the free energies, they do help to explain the shape of the free energy profiles. As expected, the average drug–water interaction energy decreases as the drug enters the bilayer and is replaced by interactions with the surrounding lipid molecules. The total interaction with the lipid tends to be stronger than that with water and this is the reason why the drugs partition into the bilayer. The barriers at the centre of the bilayer seen by each drug could be a consequence of entropic factors such as a reduction in the mobility of the lipid tails when the drug is present. However, Fig. 7 indicates that the interaction energies provide a more likely explanation. The magnitude of both the drug–water and the drug–lipid interaction decreases in the centre of the bilayer due to the lower density of atoms in this region (see Fig. 2A). Even though it is easier to create a cavity for the drug in the centre of the bilayer which would entropically favour the drug at this position, the lower interaction energies disfavours the presence of the drug. A consequence of this is that the more polar versions of the drugs see a larger barrier as they are more influenced by direct energetic interactions.

> **Figure 7**
> 
> Interpretation: A grid of six scatter plots showing energy decomposition data for both drugs with both charge parameter sets. The top row shows benzocaine with CGFF (left) and HF (right) charges; the bottom row shows phenytoin with CGFF (left) and HF (right) charges. Each subplot contains three curves: drug-lipid interaction energy (red circles), drug-water interaction energy (blue circles), and drug-total interaction energy (black circles). The x-axis shows drug position from 0 to 30 Å (0 being the bilayer center), and the y-axis shows interaction energy in kcal/mol. For all panels, drug-water interaction is strongest (most negative) in bulk water (right side) and decreases toward the bilayer center. Conversely, drug-lipid interaction is weakest at the center and increases as the drug moves toward the lipid headgroups. The drug-total curve (sum of both interactions) shows a minimum at intermediate positions and a maximum (barrier) at the bilayer center (left side of plots). The more polar HF charge sets show higher interaction energies overall compared to CGFF.
> 
> Caption: Energy decomposition. The average interaction energy of each drug with water and lipid is plotted as a function of drug position. Data points represent the average energy within each window of the umbrella sampling, while the dashed lines show the moving average of 3 data points. The sum of the drug–water and drug–lipid interactions is shown as drug-total and zeroed in bulk water (right hand side of each graph).

---

## 4. Conclusions

In this manuscript we have examined the thermodynamics and structural aspects of the partitioning of benzocaine and phenytoin into a lipid bilayer. In order to validate our simulation parameters we first calculated water–octanol partitioning coefficients which can be compared to well characterised experimental values. Different methods for determining the atomic partial charges resulted in significant differences in the polarity of the molecules, and the specific choice of charges can have a significant influence on the portioning of these drugs between these phases. While this comparison is not a complete test of drug parameters, it does offer a warning as to how much these parameters can alter quantitative simulation results. We believe that simulating water/octanol partitioning can be a useful step in validating the final choice of charges prior to their use in more complex simulations.

Both drugs were found to move favourably into the lipid bilayer, and can be expected to reside just inside the polar lipid headgroups. From this position they may either influence bilayer properties or access target membrane proteins such as the voltage-gated sodium channels to exert their anaesthetic action.

The current understanding of drug-binding in these channels considers the cytosolic activation gate as the main route of access for channel-blocking drugs [8]. However, recent demonstration of an alternative access route in a bacterial voltage-gated sodium channel, which is apparently lipid-filled, may allow for drug-passage through the lipid membrane [7]. This entrance to this so-called hydrophobic fenestration coincides with the centre of the bilayer and so also with the energy barrier seen in our results. Drugs may avoid this barrier by interacting with the protein as they reach the centre of the bilayer, but it is possible that the height of the barrier could influence the rate at which drugs enter or leave the fenestrations as well as influencing drug-access to the cytosol and from there the activation gate. The barrier in the centre of the bilayer could thus have a significant effect on the use-dependence and thus the therapeutic applications of channel-blocking drugs.

Remarkably, both drugs are able to drag water deep into the bilayer. In their most favoured position, both compounds are surrounded on one side by water molecules extending back to bulk. Even if they penetrate further into the bilayer we find that they can either pull bound water molecules with them or create a continuous chain of water molecules extending back to bulk. These situations could affect the interactions between the lipid bilayer and embedded proteins, which is an alternative mechanism of anaesthetic action [15]. While most studies of the effects of lipid bilayers on proteins focus on changes to the lateral pressure profile [15–17], none have specifically acknowledged the effect of snorkelling or water penetration generated by anaesthetic drugs. Bilayer deformation has been demonstrated to influence an archaeal potassium channel, facilitated by charged arginine residues in the voltage-sensing domain [50]. Thus, it is possible that snorkelling by local anaesthetics can perturb the interfacial interactions around ion channels and thus affect their voltage-sensing behaviour. Since we use a simple pure lipid in our simulations our results are not representative of a bacterial or eukaryotic membrane, which may be important as lipid composition has been shown to affect the bacterial sodium channel NaChBac [51].

---

## Appendix A. Supplementary data

Supplementary data to this article can be found online at http://dx.doi.org/10.1016/j.bpc.2013.12.003.

---

## References

[1] A. Scholz, Mechanisms of (local) anaesthetics on voltage-gated sodium and other ion channels, Br. J. Anaesth. 89 (2002) 52–61.

[2] S. Štolc, V. Nemeček, H. Šzicsová, Local anaesthetics: lipophilicity, charge, diffusion and site of action in isolated neuron, Eur. J. Pharmacol. 164 (2) (1989) 249–256.

[3] G.R. Strichartz, V. Sanchez, G.R. Arthur, R. Chafetz, D. Martin, Fundamental properties of local anaesthetics. ii. measured octanol:buffer partition coefficients and pKa values of clinically used drugs, Anesth. Analg. 71 (2) (1990) 158–170.

[4] L. Langerman, M. Bansinath, G.J. Grant, The partition coefficient as a predictor of local anaesthetic potency for spinal anaesthesia: evaluation of five local anaesthetics in a mouse model, Anesth. Analg. 79 (3) (1994) 490–494.

[5] L. Langerman, E. Golomb, G.J. Grant, S. Benita, Duration of spinal anaesthesia is determined by the partition coefficient of local anaesthetic, Br. J. Anaesth. 72 (4) (1994) 456–459.

[6] N.P. Franks, W.R. Lieb, Molecular and cellular mechanisms of general anaesthesia, Nature 367 (1994) 607–614.

[7] B. Hille, Local anaesthetics: hydrophilic and hydrophobic pathways for the drug-receptor reaction, J. Gen. Physiol. 69 (1977) 497–515.

[8] J. Payandeh, T. Scheuer, N. Zheng, W.A. Catterall, The crystal structure of a voltage-gated sodium channel, Nature 475 (2011) 353–358.

[9] J. Payandeh, T.M.G. El-Din, T. Scheuer, N. Zheng, W.A. Catterall, Crystal structure of a voltage-gated sodium channel in two potentially inactivated states, Nature 486 (2012) 135–139.

[10] X. Zhang, W. Ren, P. DeCaen, C. Yan, X. Tao, L. Tang, J. Wang, K. Hasegawa, T. Kumasaka, J. He, J. Wang, D.E. Clapham, N. Yan, Crystal structure of an orthologue of the NaChBac voltage-gated sodium channel, Nature 486 (2012) 130–134.

[11] E.C. McCusker, C. Bagriris, C.E. Naylor, A.R. Cole, N. D'Avanzo, C.G. Nichols, B.A. Wallace, Structure of a bacterial voltage-gated sodium channel pore reveals mechanisms of opening and closing, Nat. Commun. 3 (2012) 1102.

[12] T. Heimburg, A.D. Jackson, The thermodynamics of general anaesthesia, Biophys. J. 92 (2007) 3159–3165.

[13] H.H. Meyer, Zur theorie der alkoholnarkose, Arch. Exp. Pathol. Pharmakol. 42 (1899) 109–118.

[14] C.E. Overton, Studien über die Narkose. Zugleich ein Beitrag zur allgemeinen Pharmakologie, G. Fischer Verlag, Jena, 1901.

[15] R.S. Cantor, The lateral pressure profile in membranes: a physical mechanism of general anesthesia, Biochemistry 36 (9) (1997) 2339–2344.

[16] K. Wodzinska, A. Blicher, T. Heimburg, The thermodynamics of lipid ion channel formation in the absence and presence of anaesthetics. BLM experiments and simulations, Soft Matter 5 (2009) 3319–3330.

[17] H. Jerabek, G. Pabst, M. Rappolt, T. Stockner, Membrane-mediated effect on ion channels induced by the anaesthetic drug ketamine, J. Am. Chem. Soc. 13 (23) (2010) 7990–7997.

[18] D. Bemporad, C. Luttmann, J. Essex, Computer simulation of small molecule permeation across a lipid bilayer: dependence on bilayer properties and solute volume, size, and cross-sectional area, Biophys. J. 87 (2004) 1–13.

[19] R.D. Porasso, W.F. Drew Bennett, B.D. Oliveira-Costa, J.J. Lopez Cascales, Study of the benzocaine transfer from aqueous solution to the interior of a biological membrane, J. Phys. Chem. B 113 (29) (2009) 9988–9994.

[20] J.J.L. Cascales, S.D.O. Costa, R.D. Porasso, Thermodynamic study of benzocaine insertion into different lipid bilayers, J. Chem. Phys. 135 (2011) 135103.

[21] C. Högberg, A. Maliniak, A.P. Lyubartsev, Dynamical and structural properties of charged and uncharged lidocaine in a lipid bilayer, Biophys. Chem. 125 (2–3) (2007) 416–424 (URL www.scopus.com).

[22] C. Högberg, A.P. Lyubartsev, Effect of local anesthetic lidocaine on electrostatic properties of a lipid bilayer, Biophys. J. 94 (2) (2008) 525–531.

[23] E.H. Mojumdar, A.P. Lyubartsev, Molecular dynamics simulations of local anesthetic articaine in a lipid bilayer, Biophys. Chem. 153 (1) (2010) 27–35.

[24] L.R. Pratt, A. Pohorille, Hydrophobic effects and modeling of biophysical aqueous solution interfaces, Chem. Rev. 102 (8) (2002) 2671–2692, http://dx.doi.org/10.1021/cr000692+.

[25] L. Martínez, R. Andrade, E.G. Birgin, J.M. Martínez, Packmol: a package for building initial configurations for molecular dynamics simulations, J. Comp. Chem. 30 (2009) 2157–2164.

[26] W. Humphrey, A. Dalke, K. Schulten, VMD —visual molecular dynamics, J. Mol. Graph. 14 (1996) 33–38.

[27] R.W. Zwanzig, High temperature equation of state by a perturbation method. I. Nonpolar gases, J. Chem. Phys. 22 (8) (1954) 1420–1426.

[28] K. Vanommeslaeghe, E. Hatcher, C. Acharya, S. Kundu, S. Zhong, J. Shim, E. Darian, O. Guvench, P. Lopes, I. Vorobyov, A.D. MacKerell, CHARMM general force field: a force field for drug-like molecules compatible with the CHARMM all-atom additive biological force fields, J. Comp. Chem. 31 (4) (2010) 671–690.

[29] M.J. Frisch, et al., Gaussian 03, Revision C.02, Gaussian, Inc., Wallingford, CT, 2004.

[30] U.C. Singh, P.A. Kollman, An approach to computing electrostatic charges for molecules, J. Comp. Chem. 5 (1984) 129–145.

[31] C.H. Besler, K.M. Merz Jr., P.A. Kollman, Atomic charges derived from semiempirical methods, J. Comp. Chem. 11 (1990) 431–439.

[32] A.D. MacKerell Jr., D. Bashford, M. Bellott, R.L. Dunbrack Jr., J.D. Evanseck, M.J. Field, S. Fischer, J. Gao, H. Guo, S. Ha, D. Joseph-McCarthy, L. Kuchnir, K. Kuczera, F.T.K. Lau, C. Mattos, S. Michnick, T. Ngo, D.T. Nguyen, B. Prodhom, W.E.R. III, B. Roux, M. Schlenkrich, J.C. Smith, R. Stote, J. Straub, M. Watanabe, J. Wiórkiewicz-Kuczera, D. Yin, M. Karplus, All-atom empirical potential for molecular modeling and dynamics studies of proteins, J. Phys. Chem. B 102 (1998) 3586–3616.

[33] G. Torrie, J. Valleau, Monte Carlo free energy estimates using non-Boltzmann sampling: application to the sub-critical Lennard-Jones fluid, Chem. Phys. Lett. 28 (1974) 578–581.

[34] S. Kumar, D. Bouzida, R. Swendsen, P. Kollman, J. Rosenberg, The weighted histogram analysis method for free energy calculations on biomolecules.1. The method, J. Comput. Chem. 13 (1992) 1011–1021.

[35] B. Roux, The calculation of potential of mean force using computer simulations, Comput. Phys. Commun. 91 (1995) 275–282.

[36] A. Grossfield, Wham: the weighted histogram analysis method, http://membrane.urmc.rochester.edu/content/wham.

[37] J.C. Phillips, R. Braun, W. Wang, J. Gumbart, E. Tajkhorshid, E. Villa, C. Chipot, R.D. Skeel, L. Kale, K. Schulten, Scalable molecular dynamics with NAMD, J. Comp. Chem. 26 (2005) 1781–1802.

[38] J.B. Klauda, R.M. Venable, J.A. Freites, J.W. OConnor, D.J. Tobias, C. Mondragon-Ramirez, I. Vorobyov, A.D. MacKerell, R.W. Pastor, Update of the CHARMM all-atom additive force field for lipids: validation on six lipid types, J. Phys. Chem. B 114 (23) (2010) 7830–7843.

[39] I.S. Joung, T.E. Cheatham III, Determination of alkali and halide monovalent ion parameters for use in explicitly solvated biomolecular simulations, J. Phys. Chem. B 112 (2008) 9020–9041.

[40] S.D. Mithani, V. Bakatselou, C.N. TenHoor, J.B. Dressman, Estimation of the increase in solubility of drugs as a function of bile salt concentration, Pharm. Res. 13 (1) (1996) 163–167.

[41] V.J. Stella, S. Martodihardjo, K. Terada, V.M. Rao, Some relationships between the physical properties of various 3-acyloxymethyl prodrugs of phenytoin to structure: potential in vivo performance implications, J. Pharm. Sci. 87 (10) (1998) 1235–1241.

[42] D.J. Livingstone, M.G. Ford, J.J. Huuskonen, D.W. Salt, Simultaneous prediction of aqueous solubility and octanol/water partition coefficient based on descriptors derived from molecular structure, J. Comput. Aided Mol. Des. 15 (8) (2001) 741–752.

[43] M. Tripşa, V.E. Sahini, C. Nae, V. Vasilescu, Structure/nerve membrane effect relationships of some local anaesthetics, Gen. Physiol. Biophys. 5 (4) (1986) 371–376.

[44] C.M. Avila, F. Martínez, Thermodynamics of partitioning of benzocaine in some organic solvent/buffer and liposome systems, Chem. Pharm. Bull. 51 (3) (2003) 237–240.

[45] C. Hansch, A. Leo, D.H. Hoekman, Exploring QSAR.: Fundamentals and Applications in Chemistry and Biology, American Chemical Society, Washington D.C., 1995.

[46] C.B. Fox, J.M. Harris, Confocal Raman microscopy for simultaneous monitoring of partitioning and disordering of tricyclic antidepressants in phospholipid vesicle membranes, J. Raman Spectrosc. 41 (5) (2010) 498–507.

[47] L. Li, I. Vorobyov, A.D. MacKerell Jr., T.W. Allen, Is arginine charged in a membrane? Biophys. J. 15 (2008) L11–L13.

[48] E. Strandberg, J. Killian, Snorkeling of lysine side chains in transmembrane helices: how easy can it get? FEBS Lett. 544 (2003) 69–73.

[49] S. Dorairaj, T. Allen, On the thermodynamic stability of a charged arginine side chain in a transmembrane helix, Proc. Natl. Acad. Sci. U. S. A. 104 (2007) 4943–4948.

[50] P.J. Bond, M.S.P. Sansom, Bilayer deformation by the Kv channel voltage sensor domain revealed by self-assembly simulations, Proc. Natl. Acad. Sci. U. S. A. 104 (8) (2007) 2631–2636.

[51] N. D'Avanzo, E.C. McCusker, A. Powl, A.J. Miles, C.G. Nichols, B.A. Wallace, Differential lipid dependence of the function of bacterial sodium channels, PLoS ONE 8 (4) (2013) e61216.
