# Implementationsplan: GPU-TPMC/DSMC-Solver mit Maxwell- und CLL-Oberflächenmodellen

Technische Architektur und Entwicklungsroadmap · C++20 · NVIDIA CUDA · AMD ROCm/HIP

## 1. Projektziel und technische Entscheidungen

Wir entwickeln einen eigenständigen, dreidimensionalen GPU-beschleunigten Partikelsolver für verdünnte Gasströmungen um beliebige triangulierte Körper, insbesondere Satelliten im VLEO.

Die Geometrieoptimierung bleibt vollständig außerhalb des Projektumfangs. Der Solver erhält eine fertige Oberflächengeometrie, eine Gaskonfiguration und die Randbedingungen und berechnet daraus die aerodynamischen Kräfte, Momente und die Wärmeübertragung.

Das System soll zwei Simulationsverfahren bereitstellen:

- TPMC (Test Particle Monte Carlo): Freimolekulare Strömung ohne Gas-Gas-Kollisionen.
- DSMC (Direct Simulation Monte Carlo): Verdünnte Strömung mit stochastisch modellierten intermolekularen Kollisionen.

Beide Verfahren verwenden dieselbe Geometrie-Engine, dieselben Gas-Oberflächen-Interaktionsmodelle und dieselbe Ergebnisauswertung.

### Zentrale Architekturentscheidungen

| Bereich                             | Festlegung                                                    |
| ----------------------------------- | ------------------------------------------------------------- |
| Programmiersprache                  | C++20                                                         |
| GPU-Abstraktion                     | Kokkos                                                        |
| NVIDIA-Backend                      | CUDA                                                          |
| AMD-Backend                         | HIP / ROCm                                                    |
| CPU-Referenzbackend                 | Kokkos Serial oder OpenMP                                     |
| Build-System                        | CMake mit separaten Build-Konfigurationen                     |
| Konfigurationsdateien               | YAML                                                          |
| Geometrieformat                     | STL, intern trianguliertes Mesh                               |
| Beschleunigung der Oberflächensuche | Bounding Volume Hierarchy (BVH)                               |
| GSI-Modelle                         | Maxwell und Cercignani–Lampis–Lord (CLL)                      |
| DSMC-Gaskollisionen                 | Variable Hard Sphere (VHS), später Variable Soft Sphere (VSS) |
| Ergebnisse                          | JSON, CSV und optional HDF5                                   |
| Validierung                         | Analytische Grenzfälle und Vergleich mit SPARTA               |

Ich empfehle Kokkos statt einer reinen CUDA-Implementierung. Kokkos unterstützt CUDA für NVIDIA- und HIP für AMD-GPUs und erlaubt uns, die numerischen Kernalgorithmen weitgehend gemeinsam zu implementieren. Die GPU-Architektur wird beim Kompilieren ausgewählt.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://kokkos.org\&sz=32)

Kokkos documentation+1



Unser Ziel sind deshalb nicht zwei separat entwickelte Solver, sondern:

\\[ \boxed{ \text{Ein physikalischer Solver} + \text{zwei GPU-Backends} + \text{ein CPU-Referenzbackend} } \\]

## 2. Verbindliche Ergebnisgrößen

Jede Simulation soll zunächst genau die von dir gewünschten zentralen aerodynamischen Größen liefern.

Gesamtkraft

\\(\mathbf F\ [\mathrm N]\\)Dreidimensionaler Kraftvektor

Gesamtmoment

\\(\mathbf M\ [\mathrm{N\\,m}]\\)Moment um einen Referenzpunkt

Widerstandsbeiwert

\\(C_D\ [-]\\)Projizierte Widerstandskraft

Auftriebsbeiwert

\\(C_L\ [-]\\)Vorzeichenbehafteter Auftrieb

Momentenbeiwerte

\\(\mathbf C_M\ [-]\\)Drei dimensionslose Komponenten

Wärmeübertragungsrate

\\(\dot Q\ [\mathrm W]\\)Nettoenergieübertrag an die Oberfläche

Zusätzlich sollten wir interne Diagnosegrößen ausgeben: Anzahl injizierter und reflektierter Partikel, Gas-Gas-Kollisionen, statistische Konvergenz, Laufzeit und Partikelverluste. Diese Größen sind notwendig, um die physikalische Verlässlichkeit der sechs Hauptausgaben nachzuweisen.

### Definition der Koeffizienten

Wir definieren den dynamischen Referenzdruck als

\\[ q\_\infty=\frac12\rho\_\infty V\_\infty^2. \\]

Mit der vorgegebenen Referenzfläche \\(A\_{\mathrm{ref}}\\) und Referenzlänge \\(L\_{\mathrm{ref}}\\):

\\[ C_D=\frac{\mathbf F\cdot\mathbf e_D} {q\_\infty A\_{\mathrm{ref}}}, \\]

\\[ C_L=\frac{\mathbf F\cdot\mathbf e_L} {q\_\infty A\_{\mathrm{ref}}}, \\]

\\[ \boxed{ \mathbf C_M= \frac{\mathbf M} {q\_\infty A\_{\mathrm{ref}}L\_{\mathrm{ref}}}. } \\]

Dabei zeigt \\(\mathbf e_D\\) in die Richtung des einfallenden Gasstroms und damit in die Widerstandsrichtung. \\(\mathbf e_L\\) ist die definierte positive Auftriebsrichtung, senkrecht zu \\(\mathbf e_D\\).

Für die allgemeine dreidimensionale Anströmung mit AoA (angle of attack) und AoS (angle of sideslip) müssen wir zusätzlich eine Seitenkraftrichtung festlegen. Obwohl \\(C_Y\\) nicht zu den sechs geforderten Hauptausgaben gehört, würde ich ihn ergänzend ausgeben, damit keine Kraftkomponente verloren geht.

Die Referenzfläche und -länge müssen explizite Eingabeparameter sein, nicht aus der Bounding Box automatisch abgeleitete Werte.

## 3. Physikalische und softwaretechnische Gesamtarchitektur

Simulation Configuration

Mesh · Gasarten · Anströmung · GSI-Modell · numerische Parameter · Hardware

Geometry Engine

Trianguliertes Mesh · Normalen · Flächen · Material-IDs · BVH

GPU Particle Engine

Partikelinjektion · Partikelbewegung · Ray-Triangle-Intersection

TPMC

Kollisionsfreier Transport

DSMC

Transport + Gas-Gas-Kollisionen

Surface Interaction Engine

Maxwell

CLL

Impuls- und Energieübertrag bei jedem Oberflächenkontakt

GPU Tally + Statistical Reduction

\\(\mathbf F,\ \mathbf M,\ C_D,\ C_L,\ \mathbf C_M,\ \dot Q\\)Global und optional pro Oberflächenelement

Output and Diagnostics

JSON · CSV · HDF5 · Konvergenz · Fehlerdiagnose

Der entscheidende Softwaregrundsatz: TPMC und DSMC teilen dieselbe GSI-Implementierung. Es darf keine getrennten Maxwell- oder CLL-Implementierungen geben, die auf unterschiedlichen GPUs oder in unterschiedlichen Solvermodi abweichende Physik erzeugen.

## 4. Implementierung der Oberflächenmodelle

Die Oberflächenmodelle sind ein eigenständiges Modul. Beide arbeiten im lokalen, mit der Oberfläche mitbewegten Koordinatensystem.

Wir definieren

\\[ \mathbf c_i=\mathbf v_i-\mathbf v_w \\]

als einfallende Geschwindigkeit relativ zur Wand, mit Wandgeschwindigkeit \\(\mathbf v_w\\).

Die Oberflächennormale \\(\mathbf n\\) zeigt vom Festkörper in das Gas. Für ein einfallendes Partikel gilt daher

\\[ \mathbf c_i\cdot\mathbf n<0. \\]

### 4.1 Maxwell-Modell

Das Maxwell-Modell benötigt zwei Parameter:

\\[ f_d\in[0,1],\qquad T_w>0. \\]

Dabei ist \\(f_d\\) der Anteil diffuser Reflexionen.

Für jeden Partikelkontakt erzeugen wir eine gleichverteilte Zufallszahl \\(R\in[0,1)\\).

Falls \\(R\geq f_d\\), wird spiegelnd reflektiert:

\\[ \mathbf c_r= \mathbf c_i-2(\mathbf c_i\cdot\mathbf n)\mathbf n. \\]

Andernfalls wird diffus reflektiert.

Mit

\\[ s=\sqrt{\frac{k_BT_w}{m}} \\]

erzeugen wir in einer lokalen Orthonormalbasis \\((\mathbf t_1,\mathbf t_2,\mathbf n)\\):

\\[ \begin{aligned} c\_{r,t_1}&=sZ_1,\\\ c\_{r,t_2}&=sZ_2,\\\ c\_{r,n}&=s\sqrt{-2\ln R_1}, \end{aligned} \\]

mit unabhängigen standardnormalverteilten \\(Z_1,Z_2\\) und \\(R_1\sim\mathcal U(0,1)\\).

Der positive Normalanteil berücksichtigt die korrekte flussgewichtete diffuse Emissionsverteilung.

### 4.2 CLL-Modell

Für CLL würde ich ausdrücklich die Parameterkonvention

\\[ \alpha_n\in[0,1],\qquad \sigma_t\in[0,2] \\]

festlegen, wobei \\(\alpha_n\\) die normale Energieakkommodation und \\(\sigma_t\\) die tangentiale Impulsakkommodation beschreibt.

Die tangentiale Auslaufgeschwindigkeit lässt sich durch

\\[ \boxed{ \mathbf c\_{r,t} = (1-\sigma_t)\mathbf c\_{i,t} + s\sqrt{\sigma_t(2-\sigma_t)} \begin{pmatrix}Z_1\\\Z_2\end{pmatrix} } \\]

erzeugen.

Für den positiven Normalanteil verwenden wir zwei weitere unabhängige Standardnormalvariablen \\(Z_3,Z_4\\):

\\[ \boxed{ c\_{r,n} = \sqrt{ \left( \sqrt{1-\alpha_n}|c\_{i,n}| +s\sqrt{\alpha_n}Z_3 \right)^2 + s^2\alpha_n Z_4^2 }. } \\]

Diese Stichprobenform realisiert den klassischen translatorischen CLL-Streukern. Sie vermeidet auch das direkte numerische Auswerten der modifizierten Besselfunktion in der Normalverteilung.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://mcaneff.github.io\&sz=32)

mcaneff | My Portfolio+1



Die zwei wichtigsten Grenzfälle sind:

| \\(\alpha_n\\) | \\(\sigma_t\\) | Erwartetes Verhalten                     |
| -------------- | -------------- | ---------------------------------------- |
| 0              | 0              | Spiegelnde Reflexion                     |
| 1              | 1              | Vollständig diffuse thermische Reflexion |

Wichtig für die Vergleichbarkeit mit SPARTA: Manche Implementierungen parametrisieren die tangentiale Energieakkommodation mit \\(\alpha_t\\) statt mit der tangentialen Impulsakkommodation \\(\sigma_t\\). Es gilt

\\[ \alpha_t=\sigma_t(2-\sigma_t). \\]

Unsere Konfiguration muss diese Unterscheidung eindeutig dokumentieren. Wir dürfen beispielsweise \\(\sigma_t=0.7\\) nicht ungeprüft als `acc_t=0.7` an eine anders parametrisierte Implementierung übergeben.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.scielo.br\&sz=32)

scielo.br+1



### Gemeinsame GSI-Schnittstelle

Im C++-Code soll die Auswahl zwischen den Modellen über eine identische Schnittstelle erfolgen:

```
enum class GSIModel : uint8_t {
    Maxwell,
    CLL
};

struct SurfaceProperties {
    GSIModel model;

    double temperature_K;

    // Maxwell
    double diffuse_fraction;

    // CLL
    double alpha_n;
    double sigma_t;
};

struct ScatteringResult {
    Vec3 velocity_out;
    double internal_energy_out;
};

KOKKOS_INLINE_FUNCTION
ScatteringResult scatter_particle(
    const Vec3& velocity_in,
    const Vec3& wall_velocity,
    const Vec3& outward_normal,
    double particle_mass,
    double internal_energy_in,
    const SurfaceProperties& surface,
    RandomStream& rng
);
```

Dies ist ein Interface-Entwurf, kein bereits kompilierter Solver-Code. Der tatsächliche Typ `RandomStream` muss als GPU-kompatibler, deterministischer Zufallszahlengenerator implementiert werden.

Für die erste Version beschränken wir den Wärmeübertrag ausdrücklich auf translatorische Energie und monatomare Gasarten. Die Felder für innere Energie bleiben als Vorbereitung für N₂ und O₂ erhalten.

## 5. Berechnung von Kräften, Momenten und Wärmeübertragung

Ich würde alle Ausgaben auf einer einzigen Ereignisbilanz aufbauen.

Für den Kontakt \\(e\\) gilt:

\\[ \Delta\mathbf p_e = m_e(\mathbf v\_{i,e}-\mathbf v\_{r,e}). \\]

Der Beitrag zum Moment um den festgelegten Referenzpunkt \\(\mathbf r_0\\) ist

\\[ \Delta\mathbf l_e = (\mathbf r_e-\mathbf r_0)\times\Delta\mathbf p_e. \\]

Die translatorische Energieübertragung an eine lokal ruhend betrachtete Wand lautet

\\[ \Delta E_e = \frac12 m_e \left( |\mathbf c\_{i,e}|^2- |\mathbf c\_{r,e}|^2 \right). \\]

Mit innerer Molekülenergie kommt später die Differenz \\(E\_{\mathrm{int},i}-E\_{\mathrm{int},r}\\) hinzu.

Für den ersten Release setzen wir chemische Reaktionen, Adsorption und Desorption nicht voraus. Damit ist \\(\dot Q\\) zunächst die Nettoleistung des modellierten Gas-Wand-Energietransfers, nicht eine vollständige thermochemische Oberflächenheizrate.

### Gemeinsame Auswertung

Wir definieren für jedes Kollisionsereignis einen effektiven Ereignisratenfaktor \\(\gamma_e\\) mit der Einheit \\(\mathrm{s}^{-1}\\).

Für steady-state TPMC ist dieser Faktor das dem Testpartikel zugeordnete statistische Molekülflussgewicht. Für DSMC ergibt er sich bei konstantem Partikelgewicht aus dem Partikelgewicht geteilt durch die gesamte Auswertungszeit.

Dann gilt:

\\[ \boxed{ \mathbf F=\sum_e\gamma_e\Delta\mathbf p_e } \\]

\\[ \boxed{ \mathbf M=\sum_e\gamma_e\Delta\mathbf l_e } \\]

und

\\[ \boxed{ \dot Q=\sum_e\gamma_e\Delta E_e. } \\]

Damit verwenden beide Simulationsverfahren dieselbe Auswertungslogik, obwohl die Partikelgewichte unterschiedlich interpretiert werden.

Vorzeichenkonvention: Positive \\(\dot Q\\) bedeutet Wärmeübertragung vom Gas zur Oberfläche. Positive und negative Teilbeiträge sollen separat auswertbar sein.

### Statistische Auswertung

Für jede Hauptausgabe implementieren wir neben dem Mittelwert auch einen Schätzfehler und ein Konfidenzintervall.

Bei TPMC können unabhängige Partikelhistorien zur Varianzschätzung genutzt werden.

Bei DSMC sind zeitlich aufeinanderfolgende Messungen typischerweise korreliert. Deshalb benötigen wir Blockmittelung oder eine andere geeignete Methode zur Schätzung des statistischen Fehlers.

Die Auswertung darf nicht ausschließlich nach einer festen Partikelanzahl abbrechen. Es sollte zusätzlich eine optionale Konvergenzbedingung geben.

## 6. GPU-TPMC: Ablauf der ersten Solver-Engine

Der TPMC-Modus ist die Grundlage des Projekts.

Einzelne Partikelhistorie

1

Partikel aus der physikalisch korrekten Einlassflussverteilung erzeugen

2

Partikelposition und Geschwindigkeit auf die GPU übertragen bzw. dort generieren

3

Nächsten Schnittpunkt mit der Oberfläche mithilfe der BVH suchen

4

Falls ein Schnittpunkt existiert: Maxwell oder CLL anwenden

5

Impuls- und Energieübertrag aufsummieren

6

Reflektiertes Partikel weiterverfolgen, inklusive mehrfacher Reflexionen

7

Beim Verlassen des Simulationsgebiets die Historie abschließen

Wichtig ist die korrekte Injektion: An einer Einlassfläche müssen nicht einfach beliebige Maxwell-Geschwindigkeiten, sondern die dazugehörigen durch die Fläche gerichteten Teilchenflüsse berücksichtigt werden. Andernfalls sind die resultierenden Kräfte trotz korrektem GSI-Modell systematisch falsch.

Die BVH können wir zunächst auf der CPU erzeugen und einmalig auf die GPU übertragen. Der Partikeltransport und die Suche nach Dreiecksschnittpunkten erfolgen dann vollständig auf der GPU.

Für komplexe Oberflächen brauchen wir eine robuste Behandlung von sehr kurzen Wiederauftreffstrecken, Dreieckskanten und mehrfachen Reflexionen.

## 7. GPU-DSMC: Erweiterung um intermolekulare Kollisionen

DSMC erweitert die TPMC-Partikelbewegung um ein räumliches Kollisionsgitter.

Die wichtigste algorithmische Erweiterung ist ein GPU-basierter No-Time-Counter-Algorithmus (NTC) nach Bird.

Für jede Simulationszelle:

1. Partikel nach Zellzugehörigkeit gruppieren.
2. Kandidaten für intermolekulare Kollisionen bestimmen.
3. Kollisionspartner zufällig auswählen.
4. Kollision anhand der relativen Geschwindigkeit und des verwendeten Querschnittmodells akzeptieren oder verwerfen.
5. Nach akzeptierter Kollision die Geschwindigkeiten unter Erhaltung von Impuls und Energie aktualisieren.

Für identisch gewichtete Simulationspartikel derselben Gasart ist die erwartete Zahl von Kollisionskandidaten näherungsweise

\\[ N\_{\mathrm{cand}} = \frac{N_c(N_c-1)}{2} \frac{w\\,(\sigma_Tg)\_{\max}\Delta t}{V_c}. \\]

Dabei sind \\(N_c\\) die Anzahl der Simulationspartikel in der Zelle, \\(w\\) deren physikalisches Partikelgewicht, \\(V_c\\) das Gasvolumen der Zelle und \\(g\\) die Relativgeschwindigkeit.

Ein Kandidat wird dann mit der Wahrscheinlichkeit

\\[ P\_{\mathrm{accept}} = \frac{\sigma_T(g)g}{(\sigma_Tg)\_{\max}} \\]

akzeptiert.

Für Gasgemische muss dieser Algorithmus auf die jeweiligen Artenpaare und deren Kollisionsparameter erweitert werden. Außerdem benötigt NTC eine korrekte Behandlung von veränderlichen Kollisionsratenobergrenzen und der diskreten Kandidatenanzahl.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://simulation.ist.fraunhofer.de\&sz=32)

DSMC & PIC-MC Documentation+1



### DSMC-Auflösungsbedingungen

Das erste DSMC-Gitter sollte ein kartesisches Hintergrundgitter sein.

Als Ausgangspunkt zur Wahl der numerischen Parameter verwenden wir:

\\[ \Delta x\_{\mathrm{cell}}\lesssim\lambda \\]

und

\\[ \Delta t\ll\tau\_{\mathrm{coll}}, \\]

wobei \\(\tau\_{\mathrm{coll}}\\) die mittlere Zeit zwischen intermolekularen Kollisionen bezeichnet.

Zusätzlich muss der Zeitschritt klein genug sein, damit die Partikelbewegung innerhalb des Kollisionsgitters zeitlich hinreichend aufgelöst wird.

Diese Regeln sind keine automatische Genauigkeitsgarantie. Zellgröße, Zeitschritt und Zahl der Simulationspartikel pro Zelle müssen über Konvergenzstudien kontrolliert werden.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.sciencedirect.com\&sz=32)

ScienceDirect



Für Körper im Gitter brauchen wir außerdem eine Behandlung angeschnittener Zellen: Ihr für Gas-Gas-Kollisionen wirksames Volumen ist nicht automatisch das vollständige kartesische Zellvolumen.

### Unterstützte Gasarten

Ich würde die physikalischen Modelle gestaffelt freigeben:

| Ausbaustand      | Gasarten             | Interne Energie               |
| ---------------- | -------------------- | ----------------------------- |
| TPMC-Basis       | O, N, He             | Nur Translation               |
| DSMC-Basis       | O, N, He             | Elastische Kollisionen        |
| DSMC-Erweiterung | Zusätzlich N₂ und O₂ | Rotation und später Vibration |

Die molekularen internen Freiheitsgrade sollten nicht stillschweigend ignoriert werden, wenn wir eine physikalisch vollständige Heizrate für ein Gasgemisch ausgeben wollen.

Für die erste DSMC-Version wird VHS implementiert. VSS sowie Larsen–Borgnakke für die Energieumverteilung zwischen translatorischen und inneren Freiheitsgraden folgen als separate Erweiterungen.

## 8. NVIDIA- und AMD-GPU-Konfiguration

Wir erhalten zwei getrennte GPU-Builds aus demselben Quellcode.

NVIDIA

CUDA-Backend

Compiler: CUDA `nvcc` über Kokkos `nvcc_wrapper`

Beispiel: NVIDIA A100

`Kokkos_ARCH_AMPERE80`

AMD

HIP-Backend

Compiler: ROCm `hipcc`

Beispiel: AMD MI300X

`Kokkos_ARCH_AMD_GFX942`

Weitere Beispiele sind `Kokkos_ARCH_HOPPER90` für NVIDIA H100, `Kokkos_ARCH_AMD_GFX90A` für AMD MI250 und `Kokkos_ARCH_AMD_GFX1100` für bestimmte Radeon-RX-7900-Modelle. Die konkrete ROCm-Unterstützung hängt auch von der verwendeten Treiber- und Softwareversion ab.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://kokkos.org\&sz=32)

Kokkos documentation+1



### CMake-Grundstruktur

Wir binden eine fest versionierte Kopie von Kokkos in `third_party/kokkos` ein.

```
cmake_minimum_required(VERSION 3.27)

project(VLEOParticleSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third_party/kokkos)

add_subdirectory(src/core)
add_subdirectory(src/geometry)
add_subdirectory(src/particles)
add_subdirectory(src/gsi)
add_subdirectory(src/dsmc)
add_subdirectory(src/tally)

add_executable(vleo_solver src/main.cpp)

target_link_libraries(vleo_solver PRIVATE
    vleo_core
    vleo_geometry
    vleo_particles
    vleo_gsi
    vleo_dsmc
    vleo_tally
    Kokkos::kokkos
)
```

Die Unterbibliotheken werden im Projekt definiert und ebenfalls mit Kokkos verknüpft. Sie dürfen keine versehentlichen Abhängigkeiten von CUDA-only- oder HIP-only-Datentypen besitzen.

### NVIDIA-Build

Für eine A100 als Beispiel:

```

cmake -S . -B build/nvidia -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$PWD/third_party/kokkos/bin/nvcc_wrapper" \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON

cmake --build build/nvidia --parallel

```

### AMD-Build

Für eine AMD MI300X:

```

cmake -S . -B build/amd -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ARCH_AMD_GFX942=ON

cmake --build build/amd --parallel

```

Die Befehle setzen eine korrekt installierte CUDA- beziehungsweise ROCm-Toolchain voraus. Zudem muss die verwendete Kokkos-Version mit Compiler, Zielarchitektur und C++20 kompatibel sein.

Die beiden Builds nutzen dieselben Quellen und Schnittstellen, aber unterschiedliche GPU-Binaries. Das ist wichtig: Ein zur Laufzeit ausgewählter YAML-Eintrag allein kann ein NVIDIA-Binary nicht in ein AMD-Binary verwandeln.

Ein dritter CPU-Build dient der Entwicklung und als numerische Referenz.

## 9. Beispiel für die Laufzeitkonfiguration

Neben den GPU-Build-Konfigurationen benötigen wir physikalische Konfigurationen.

Ich würde die Trennung so gestalten:

```
configs/
├── platforms/
│   ├── nvidia.yaml
│   ├── amd.yaml
│   └── cpu.yaml
├── simulations/
│   ├── tpmc_maxwell.yaml
│   ├── tpmc_cll.yaml
│   ├── dsmc_maxwell.yaml
│   └── dsmc_cll.yaml
└── materials/
    └── example_surface.yaml
```

Ein beispielhaftes Simulationsprofil:

```

simulation:
  name: vleo_reference
  mode: tpmc          # tpmc | dsmc
  random_seed: 12345

geometry:
  file: geometry/satellite.stl
  units: m

reference:
  area_m2: 1.0
  length_m: 1.0
  moment_center_m: [0.0, 0.0, 0.0]
  lift_axis_body: [0.0, 0.0, 1.0]

gas:
  number_density_m3: 1.0e14
  translational_temperature_K: 900.0
  bulk_velocity_body_mps: [-7800.0, 0.0, 0.0]

  # Zahlenanteile, nur illustrative Werte
  species:
    O: 0.80
    N: 0.15
    He: 0.05

surface:
  model: cll           # maxwell | cll
  temperature_K: 300.0

  maxwell:
    diffuse_fraction: 0.9

  cll:
    alpha_n: 0.85
    sigma_t: 0.65

tpmc:
  particle_histories: 2000000
  max_reflections: 100

dsmc:
  collision_model: VHS
  timestep_mode: automatic
  cell_size_mode: automatic
  particles_per_cell_target: 30

statistics:
  confidence_level: 0.95
  report_uncertainty: true

output:
  directory: results/
  format: hdf5
  save_surface_fields: true

```

Die Werte sind lediglich für die Konfigurationsstruktur gedacht und keine kalibrierte VLEO-Referenzatmosphäre. Auch `automatic` für Zeitschritt und Zellgröße ist eine geplante Solverfunktion, keine bereits implementierte Berechnung.

Die Plattformdateien enthalten dagegen Backend-Informationen und Gerätepräferenzen. Der Solver prüft beim Start, ob die Hardware mit dem kompilierten Backend übereinstimmt.

## 10. Datenstrukturen und Verzeichnisarchitektur

Die rechenintensiven Partikeldaten sollten als Structure of Arrays (SoA) organisiert werden.

Statt eine große C++-Struktur pro Partikel zu speichern, verwalten wir Positionen, Geschwindigkeiten, Gasart, Gewicht und Zustand in getrennten GPU-Arrays.

Beispielsweise:

```
struct ParticleStorage {
    Kokkos::View<double*> x, y, z;
    Kokkos::View<double*> vx, vy, vz;

    Kokkos::View<double*> weight;
    Kokkos::View<uint32_t*> species;
    Kokkos::View<uint64_t*> particle_id;

    Kokkos::View<uint8_t*> status;
};
```

Diese Speicherstruktur erleichtert koaleszierte Speicherzugriffe, die Sortierung von Partikeln nach DSMC-Zellen und die effiziente Verarbeitung auf NVIDIA und AMD.

Für die erste Version würde ich Positions- und Geschwindigkeitsdaten sowie alle physikalischen Summationen in Double Precision führen. Gemischte Präzision kann später anhand quantifizierter Fehler untersucht werden.

### Vorgeschlagene Repository-Struktur

```
vleo-particle-solver/
├── CMakeLists.txt
├── CMakePresets.json
├── cmake/
├── configs/
├── third_party/
│   └── kokkos/
├── include/
│   └── vleo/
├── src/
│   ├── core/
│   ├── geometry/
│   ├── particles/
│   ├── gsi/
│   │   ├── Maxwell.cpp
│   │   └── CLL.cpp
│   ├── tpmc/
│   ├── dsmc/
│   │   ├── CollisionCells.cpp
│   │   ├── VHS.cpp
│   │   └── NTC.cpp
│   ├── tally/
│   ├── io/
│   └── main.cpp
├── tests/
│   ├── unit/
│   ├── physics/
│   ├── regression/
│   └── performance/
├── benchmarks/
├── examples/
├── scripts/
└── docs/
```

Die physikalischen und GPU-spezifischen Komponenten sollen bewusst entkoppelt sein. Beispielsweise darf die Berechnung von \\(\mathbf F\\) und \\(\dot Q\\) nicht direkt mit dem CLL-Modell verwoben werden.

## 11. Abhakbarer Implementierungsplan

Status: [ ] offen · [-] in Arbeit · [x] abgeschlossen · [!] blockiert.

Eine Aufgabe wird erst abgehakt, wenn Implementierung, automatisierter Test und genannte Dokumentation vorhanden sind. Die Meilensteine M0 bis M5 werden sequenziell bearbeitet; der nächste beginnt erst nach bestandenem Gate. Physikänderungen benötigen einen dokumentierten Referenzfall.

### Meilensteinübersicht

Die Arbeitspakete sind sequenziell angeordnet. Ein Meilenstein gilt erst dann als abgeschlossen, wenn seine Tests erfolgreich sind.

M0 – Projektbasis und CPU-Referenz

CMake, Kokkos, Konfigurationsparser, SI-Einheiten, Test-Framework, reproduzierbare Zufallszahlen und grundlegende Geometrieoperationen implementieren.

Abnahme: CPU-Build, NVIDIA-Build und AMD-Build starten mit identischer Konfiguration; erste Vektor- und RNG-Tests bestehen.

M1 – GPU-Geometrie und Partikeltransport

STL-Import, Normalenprüfung, BVH, Partikelinjektion, Schnittpunktberechnung, Boundary-Handling und Mehrfachreflexionen.

Abnahme: Analytisch bekannte Strahl-Dreieck-Schnittpunkte und konservative Partikelbilanzen werden auf CPU, CUDA und HIP korrekt reproduziert.

M2 – Vollständiger TPMC mit Maxwell und CLL

Beide Streukerne implementieren, Impuls- und Energietallies ergänzen, \\(\mathbf F,\mathbf M,C_D,C_L,\mathbf C_M,\dot Q\\) berechnen und statistische Unsicherheiten ausgeben.

Abnahme: Maxwell- und CLL-Grenzfälle stimmen mit analytischen Verteilungen überein; alle sechs Ergebnisgrößen sind verfügbar.

M3 – DSMC-Gaskollisionen

Kartesisches Kollisionsgitter, GPU-Zellsortierung, NTC-Verfahren, VHS-Kollisionen und Zellvolumenbehandlung nahe Oberflächen implementieren.

Abnahme: Gas-Gas-Kollisionen erhalten Impuls und Energie; DSMC konvergiert im freimolekularen Grenzfall gegen TPMC.

M4 – Gasgemische, innere Energie und Validierung

Mehrkomponenten-Kollisionen, N₂/O₂, Rotations- und gegebenenfalls Vibrationsenergie, vollständiger zugehöriger Wärmeübertrag und Referenzfälle implementieren.

Abnahme: Energiehaushalt und Ergebnisse werden gegen analytische Referenzen sowie geeignete SPARTA-Fälle überprüft.

M5 – GPU-Performance und Release

Speicherzugriffe, Kernel-Aufrufe, Partikelsortierung, Reduktionen und BVH-Traversierung optimieren. Automatisierte Tests auf beiden GPU-Backends etablieren.

Abnahme: Wissenschaftliche Referenztests bestehen auf NVIDIA und AMD; Performanceberichte und Konvergenznachweise liegen vor.

### Detaillierte Checkliste

#### M0 – Projektbasis und CPU-Referenz

- [x] **M0.1** Verzeichnisstruktur aus Abschnitt 10 anlegen.
- [x] **M0.2** CMake mit C++20, Kokkos, Unterbibliotheken und CTest einrichten.
- [x] **M0.3** Kokkos- und Testframework-Versionen festlegen und dokumentieren. → Kokkos 5.2.2, yaml-cpp 0.9.0, Catch2 3.8.1 (`docs/dependencies.md`, `third_party/VERSIONS.md`)
- [x] **M0.4** Presets für CPU-Debug, CPU-Release, CUDA und HIP anlegen. → `CMakePresets.json`
- [x] **M0.5** Git-Commit, Compiler, Build-Typ, Kokkos-Version und Backend ausgeben. → `vleo_solver --version`, `run_metadata.json` (Git-Stand wird bei jedem Build neu erfasst; außerhalb eines Git-Checkouts `unknown`)
- [x] **M0.6** GPU-kompatiblen Vec3-Typ und Koordinatentransformationen testen. → `test_vec3`, `test_frames`, `test_geometry` (Host und Device)
- [x] **M0.7** SI-Einheiten sowie Achsen-, Normalen- und Vorzeichenkonventionen dokumentieren. → `docs/conventions.md`
- [x] **M0.8** YAML-Schema implementieren und vollständig validieren. → `docs/configuration.md`, `src/io/ConfigParser.cpp`
- [x] **M0.9** Aufgelöste Konfiguration einschließlich Defaultwerten ausgeben. → `resolved_config.yaml` (wieder einlesbar, Round-Trip getestet) und `run_metadata.json`
- [x] **M0.10** Positive und negative Konfigurationstests ergänzen. → `test_config`
- [x] **M0.11** Counter-basierten RNG aus Seed, Partikel-, Ereignis- und Stream-ID implementieren. → Philox4x32-10, `include/vleo/core/Random.hpp`
- [x] **M0.12** RNG auf Determinismus, Verteilung und numerische Randfälle testen. → `test_rng`

**Gate M0**

- [!] CPU-, CUDA- und HIP-Build kompilieren auf dokumentierten Toolchains. → CPU (`cpu-debug`, `cpu-release`, GCC 15.2) und HIP (`hip-gfx1100`, ROCm/HIP 7.1, Radeon 740M) erfüllt, jeweils 20/20 Tests; CUDA blockiert: keine NVIDIA-GPU/Toolchain verfügbar (`docs/building.md`, Abschnitt „Verified toolchains“).
- [x] Alle M0-Tests laufen über einen dokumentierten CTest-Befehl. → `ctest --preset <preset>` (20 Tests)
- [x] Ein Smoke-Test liest eine Konfiguration und schreibt vollständige Metadaten. → `smoke_run` + `smoke_metadata`

#### M1 – GPU-Geometrie und Partikeltransport

- [x] **M1.1** Binäres und ASCII-STL mit Einheitenumrechnung importieren. → `geometry/Stl.hpp`; Material-ID je `solid`-Block
- [x] **M1.2** Degenerierte Dreiecke, Orientierung, Geschlossenheit und nicht-manifold Kanten prüfen. → `geometry/Mesh.hpp`: Vertex-Welding, Kantenklassifikation, Komponenten, Orientierung per Volumenvorzeichen; offene Flächen nur mit `geometry.require_closed: false` (zweiseitig)
- [x] **M1.3** Flächen, Normalen, Bounding Box und Material-IDs GPU-tauglich speichern. → `DeviceMesh` (SoA in BVH-Reihenfolge, Rückabbildung über `source_index`)
- [x] **M1.4** Mesh-Tests für Dreieck, Würfel, Skalierung und Fehlerfälle ergänzen. → `test_stl`, `test_mesh`
- [x] **M1.5** Robusten Ray-Triangle-Test mit dokumentierter Epsilon-Strategie implementieren. → Watertight-Algorithmus (Woop et al. 2013), FMA-frei; `docs/geometry.md` §3
- [x] **M1.6** CPU-BVH-Builder mit flachen GPU-Knotenarrays implementieren. → Binned SAH, DFS-Layout, Tiefe ≤ 64
- [x] **M1.7** Iterative GPU-BVH-Traversierung implementieren. → Stack-basiert, konservativer Box-Test (`kBoxSlack`)
- [x] **M1.8** Selbsttreffer, Kanten- und Ecktreffer robust behandeln. → Kanten inklusiv + deterministischer Tie-Break; Ursprungs-Offset 1e-9·L plus Ausschluss des zuletzt getroffenen Dreiecks
- [x] **M1.9** BVH gegen Brute-Force-Suche auf zufälligen Meshes testen. → `test_ray`: identisches Dreieck und bitgleiches t
- [x] **M1.10** SoA-Partikelspeicher vollständig implementieren. → `particles/ParticleStorage.hpp`
- [x] **M1.11** Simulationsgebiet und Einlassflächen konfigurierbar erzeugen. → `domain.margin_m` oder `domain.box_min_m/box_max_m`; Einlass über alle 6 Flächen
- [x] **M1.12** Flussgewichtete Einlassverteilung statistisch validieren. → `test_inflow`: KS-Test, Mittelwert, χ² der Kanäle, Tangentialmomente
- [x] **M1.13** Transport bis Oberflächentreffer oder Gebietsaustritt implementieren. → `particles/Transport.hpp`
- [x] **M1.14** Mehrfachreflexionen mit Obergrenze und Abbruchdiagnose unterstützen. → Status `Truncated`, Zähler, max. Interaktionen je Historie
- [x] **M1.15** Zähler und Test für die vollständige Partikelbilanz implementieren. → `TransportCounters::balanced()`, `test_transport`

**Gate M1**

- [x] Analytische Strahltests bestehen auf allen verfügbaren Backends. → CPU Serial, OpenMP und HIP (Radeon 740M); CUDA nicht verfügbar (siehe Gate M0)
- [x] BVH und Brute Force liefern innerhalb dokumentierter Toleranzen denselben Treffer. → Toleranz: exakt (gleiches Dreieck, bitgleiches t), `docs/geometry.md` §4
- [x] Der Würfel-Testfall endet ohne verlorene oder doppelt gezählte Partikel. → 400 000 Historien, alle `Exited`; Trefferquote = analytische freimolekulare Stoßrate innerhalb 5σ

#### M2 – Vollständiger TPMC-Solver

- [x] **M2.1** Gemeinsame GSI-Datentypen und Scatter-Schnittstelle implementieren. → `gsi/Scatter.hpp`: `SurfaceProperties`, `ScatteringResult`, `scatter_particle()` (Header-only, Device)
- [x] **M2.2** Maxwell-Modell mit spiegelndem und diffusem Zweig implementieren.
- [x] **M2.3** CLL mit dokumentierter alpha-n-/sigma-t-Konvention implementieren. → `docs/tpmc.md` §1 (α_n Normalenergie, σ_t Tangentialimpuls, α_t = σ_t(2−σ_t))
- [x] **M2.4** Wandgeschwindigkeit und auswärts gerichtete Reflexion korrekt behandeln. → Wandsystem c = v − v_w; offene Zufallszahl bzw. Neuziehung garantiert c_r·n > 0
- [x] **M2.5** Impuls-, Drehimpuls- und Energietransfer je Ereignis implementieren. → `Transport.hpp`, Ereignisbilanz
- [x] **M2.6** TPMC-Ereignisgewicht herleiten, implementieren und dokumentieren. → γ = R/N mit tatsächlich gerechneten Historien, `docs/tpmc.md` §3
- [x] **M2.7** Hierarchische Tallies für Kraft, Moment und Wärme implementieren. → Register je Historie → deterministische Batch-Reduktion → Chan-Merge (keine globalen Atomics)
- [x] **M2.8** Strömungsachsen sowie C_D, C_L, C_Y und C_M robust berechnen. → Achsen aus validierter Konfiguration, SE aus projizierten Historienimpulsen
- [x] **M2.9** Globale und dreiecksweise Ergebnisse aus derselben Bilanz erzeugen. → gleiche Ereigniswerte; Σ Dreiecke = global (Test 1e-10)
- [x] **M2.10** Batches für Mittelwert, Standardfehler und Konfidenzintervall implementieren.
- [x] **M2.11** Optionalen Abbruch über den Fehler einer Zielgröße implementieren. → `tpmc.convergence`
- [x] **M2.12** JSON-, CSV- und optional HDF5-Ausgabe implementieren. → `results.json`, `results.csv`, `surface_fields.csv`; HDF5 bewusst nicht im Build (wird mit klarer Meldung abgelehnt)
- [x] **M2.13** Maxwell-Spiegelgrenzfall und null Wärmeübertrag testen.
- [x] **M2.14** Vollständig diffuse Maxwell-Verteilungen testen.
- [x] **M2.15** CLL-Grenzfälle (0,0) und (1,1) testen.
- [x] **M2.16** Thermisches Gleichgewicht auf Netto-Wärmefluss null testen.
- [x] **M2.17** Symmetrischen Körper auf verschwindende Quergrößen testen.
- [x] **M2.18** Flache Platte gegen analytische Freimolekulartheorie testen. → Maxwell (alle Kraftkomponenten, Q̇) und CLL (Tangentialkraft, Q̇); mit 4·10⁸ Historien Abweichung ≤ 2,7·10⁻⁵ relativ

**Gate M2 – Release v0.1-tpmc**

- [x] Alle sechs Hauptgrößen und ihre Unsicherheiten stehen im Ergebnis-JSON. → `smoke_results`
- [x] Alle Physiktests bestehen mit vorab dokumentierten Toleranzen. → `docs/tpmc.md` §7
- [x] Je ein Maxwell- und CLL-Beispiellauf ist reproduzierbar. → `repro_tpmc_maxwell_*`, `repro_tpmc_cll_*` (bitgleich)
- [-] CPU-, CUDA- und HIP-Ergebnisse stimmen numerisch beziehungsweise statistisch überein. → CPU ↔ HIP numerisch gleich (≤ 9·10⁻¹⁴ relativ); CUDA nicht verfügbar (siehe Gate M0)

#### M3 – DSMC mit elastischen VHS-Kollisionen

- [ ] **M3.1** Zeitschleife für Transport, Wandkontakt, Binning, Kollision und Sampling implementieren.
- [ ] **M3.2** Kartesisches Kollisionsgitter samt Position-zu-Zelle-Abbildung implementieren.
- [ ] **M3.3** GPU-Binning mittels Histogramm und Prefix-Sum implementieren und Eindeutigkeit prüfen.
- [ ] **M3.4** VHS-Querschnitt und NTC-Kandidatenzahl implementieren.
- [ ] **M3.5** Partner ohne Selbstkollision auswählen und Akzeptanzentscheidung implementieren.
- [ ] **M3.6** Nachkollisionsgeschwindigkeiten isotrop im Schwerpunktsystem erzeugen.
- [ ] **M3.7** Zellweise Obergrenze für sigma mal g sicher aktualisieren.
- [ ] **M3.8** Akzeptierte und verworfene Kollisionen diagnostizieren.
- [ ] **M3.9** Gasvolumenanteile angeschnittener Zellen festlegen und validieren.
- [ ] **M3.10** Kleine Volumenanteile sicher behandeln und schlechte Auflösung melden.
- [ ] **M3.11** Einlauf- und Samplingphase trennen.
- [ ] **M3.12** Blockmittelung für korrelierte DSMC-Zeitreihen implementieren.
- [ ] **M3.13** Auflösungsdiagnosen für Zelle, Zeitschritt und Partikelzahl ausgeben.
- [ ] **M3.14** Binäre Kollisionen auf Impuls- und Energieerhaltung testen.
- [ ] **M3.15** Homogenes Gas und Kollisionsrate gegen Referenzfälle testen.
- [ ] **M3.16** Verdünnungsstudie DSMC gegen TPMC durchführen.
- [ ] **M3.17** Zellgrößen-, Zeitschritt- und Partikelzahl-Konvergenz getrennt dokumentieren.

**Gate M3 – Release v0.2-dsmc**

- [ ] Konservative Kollisions- und Gleichgewichtstests bestehen.
- [ ] Angeschnittene Zellvolumina sind analytisch validiert.
- [ ] Der freimolekulare DSMC-Grenzfall stimmt statistisch mit TPMC überein.
- [ ] Ein monatomarer DSMC-Beispiellauf ist reproduzierbar.

#### M4 – Gasgemische, innere Energie und Validierung

- [ ] **M4.1** Speziesdatenbank mit Masse, VHS-Parametern, Einheiten und Quellen anlegen.
- [ ] **M4.2** Artenpaar-Kollisionen und unterschiedliche Massen implementieren.
- [ ] **M4.3** Gemischinjektion und artspezifische Diagnosen implementieren.
- [ ] **M4.4** O/N/He-Mischung gegen homogene Referenzfälle testen.
- [ ] **M4.5** Datenmodell für Rotations- und Vibrationsenergie festlegen.
- [ ] **M4.6** Larsen-Borgnakke für Rotation implementieren und validieren.
- [ ] **M4.7** N2 und O2 erst nach bestandenen Rotationstests freischalten.
- [ ] **M4.8** Wärmebilanz um innere Energie erweitern.
- [ ] **M4.9** Vibrationsmodell und VSS als getrennte Erweiterungen validieren.
- [ ] **M4.10** Maxwell-, CLL- und DSMC-Referenzfälle in SPARTA definieren.
- [ ] **M4.11** sigma-t-/alpha-t-Übersetzung dokumentieren.
- [ ] **M4.12** Ergebnisse automatisiert gegen SPARTA vergleichen.
- [ ] **M4.13** Akzeptanzgrenzen vor der Auswertung festlegen.
- [ ] **M4.14** Eingaben, Versionen und Berichte unter validation archivieren.

**Gate M4**

- [ ] Gemischkollisionen erhalten Gesamtimpuls und Gesamtenergie.
- [ ] Die Wärmebilanz schließt alle aktivierten Freiheitsgrade ein.
- [ ] SPARTA-Fälle bestehen innerhalb der definierten Toleranzen.
- [ ] Unterstützte Kombinationen sind in einer Capability-Matrix dokumentiert.

#### M5 – Performance, Automatisierung und Release

- [ ] **M5.1** Repräsentative TPMC-/DSMC-Benchmarks und Baseline festlegen.
- [ ] **M5.2** CUDA- und HIP-Profile archivieren und dominante Kernel identifizieren.
- [ ] **M5.3** BVH, DSMC-Binning und Tallies einzeln optimieren.
- [ ] **M5.4** Nach jeder Optimierung Physiktests und Vorher-/Nachher-Messung ausführen.
- [ ] **M5.5** Gemischte Präzision nur nach Fehler- und Performanceanalyse einführen.
- [ ] **M5.6** CI für Formatierung, Analyse, CPU-Build sowie Unit- und Physiktests einrichten.
- [ ] **M5.7** Regelmäßige CUDA- und HIP-Pipelines auf echter Hardware einrichten.
- [ ] **M5.8** Sanitizer-Läufe für den CPU-Build ergänzen.
- [ ] **M5.9** Toolchain-Matrix und Benutzer-, Methoden- und Validierungsdokumentation fertigstellen.
- [ ] **M5.10** Beispiele mit festen Seeds automatisiert prüfen.
- [ ] **M5.11** Lizenz, Changelog, Versionsschema und Release-Checkliste ergänzen.

**Gate M5 – Release v1.0**

- [ ] CPU-, CUDA- und HIP-Regressionstests bestehen auf dokumentierten Plattformen.
- [ ] Validierungsberichte und Konvergenzstudien sind versioniert und reproduzierbar.
- [ ] Keine offene Abweichung betrifft unterstützte Physik oder eine Hauptausgabe.
- [ ] Release-Artefakte sind aus einem sauberen Checkout reproduzierbar.

## 12. Physikalische Validierungsstrategie

Die Validierung sollte von Beginn an Teil der Entwicklung sein, nicht erst eine abschließende Aufgabe.

| Testfall                                          | Erwartetes Ergebnis                                                                      | Betroffene Komponenten        |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------- |
| Spiegelnde Reflexion an ebener Wand               | Einfalls- und Ausfallswinkel gleich; kein translatorischer Wärmeübertrag an ruhende Wand | Maxwell, Kräfte, \\(\dot Q\\) |
| Diffuse Reflexion                                 | Korrekte Maxwell-Wandflussverteilung                                                     | Maxwell                       |
| CLL mit \\(\alpha_n=0,\sigma_t=0\\)               | Spiegelnde Reflexion                                                                     | CLL                           |
| CLL mit \\(\alpha_n=1,\sigma_t=1\\)               | Vollständig diffuse Reflexion                                                            | CLL                           |
| Isothermes Gas im Gleichgewicht mit ruhender Wand | Kein mittlerer Nettoenergieübertrag                                                      | Maxwell, CLL, \\(\dot Q\\)    |
| Symmetrischer Körper in symmetrischer Anströmung  | Entsprechende Querkräfte und Momente verschwinden statistisch                            | \\(\mathbf F,\mathbf M\\)     |
| Flache Platte im freimolekularen Regime           | Vergleich mit analytischer Freimolekulartheorie                                          | TPMC, \\(C_D,C_L\\)           |
| Elastische Gas-Gas-Kollision                      | Impuls- und Energieerhaltung                                                             | DSMC                          |
| Homogenes Gas im Gleichgewicht                    | Stationäre Maxwell-Verteilung                                                            | DSMC                          |
| Verdünnungsgrenzfall                              | DSMC nähert sich TPMC                                                                    | Beide Solvermodi              |

### Vergleich mit SPARTA

SPARTA ist ein geeigneter unabhängiger Referenzsolver. Die Dokumentation beschreibt bereits diffuse und CLL-Oberflächenmodelle sowie GPU-Beschleunigung über Kokkos.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://sparta.github.io\&sz=32)

sparta.github.io+1



Für die Validierung sollten identische Testfälle in beiden Solvern verwendet werden.

Dabei genügt es nicht, nur \\(C_D\\) zu vergleichen. Wir müssen auch die Kräfte und Momente in allen drei Raumrichtungen, die Wärmeübertragung sowie die statistischen Unsicherheiten vergleichen.

Numerische Parameterkonvergenz und statistische Konvergenz müssen separat nachgewiesen werden.

Für CUDA und HIP sollten wir dagegen keine bitweise identischen Resultate verlangen: Unterschiedliche Gleitkommaoperationsreihenfolgen und parallele Reduktionen können geringfügige Unterschiede verursachen. Entscheidend ist die Übereinstimmung innerhalb definierter numerischer und statistischer Toleranzen.

## 13. Reproduzierbarkeit und GPU-Optimierung

Drei Implementierungsdetails sind besonders wichtig.

Zufallszahlengenerierung: Ich würde einen counter-basierten Generator wie Philox vorsehen. Der Zufallszustand wird aus Seed, Partikel-ID, Kollisionsindex und einem separaten Stream-Index abgeleitet. Das erleichtert reproduzierbare Simulationen über unterschiedliche GPU-Ausführungsreihenfolgen hinweg.

Reduktionen: Die Berechnung von Kräften, Momenten und Wärmeübertragung sollte zunächst mit GPU-lokalen Teilsummen und hierarchischen Reduktionen realisiert werden. Globale atomare Additionen bei jedem Oberflächenkontakt können sowohl die Performance als auch die Reproduzierbarkeit beeinträchtigen.

Partikelorganisation: Für DSMC müssen Partikel effizient nach Kollisionszellen gruppiert werden. Prefix-Sums und GPU-basierte Sortierungs- beziehungsweise Binning-Verfahren sind dafür zentral. Für TPMC ist dagegen insbesondere die BVH-Durchlaufleistung maßgeblich.

Wir sollten den Erfolg deshalb nicht nur an der Zahl verarbeiteter Partikel pro Sekunde messen, sondern an der Rechenzeit, die nötig ist, um einen vorgegebenen statistischen Fehler in \\(C_D\\), \\(\mathbf M\\) oder \\(\dot Q\\) zu erreichen.

## 14. Gewünschtes Ausgabeformat

Ich würde pro Simulationslauf eine zentrale Ergebnisdatei mit SI-Einheiten, Bezugsgrößen, Simulationsmetadaten und Qualitätsindikatoren erzeugen.

Beispielstruktur:

```
{
  "simulation": {
    "mode": "tpmc",
    "backend": "cuda",
    "gsi_model": "cll"
  },
  "reference": {
    "area_m2": 1.0,
    "length_m": 1.0,
    "moment_center_m": [0.0, 0.0, 0.0]
  },
  "results": {
    "force_N": [0.0, 0.0, 0.0],
    "moment_Nm": [0.0, 0.0, 0.0],
    "C_D": 0.0,
    "C_L": 0.0,
    "C_M": [0.0, 0.0, 0.0],
    "heat_rate_W": 0.0
  },
  "statistics": {
    "confidence_level": 0.95,
    "converged": false
  }
}
```

Die Nullen sind lediglich Platzhalter zur Darstellung des Datenformats, keine berechneten Ergebnisse.

Die tatsächlich gespeicherten Daten sollen zusätzlich die statistischen Fehler aller sechs Hauptgrößen, Simulationsparameter, Laufzeiten und den verwendeten Git-Commit enthalten.

Bei aktivierter Oberflächenauswertung speichern wir außerdem die lokalen Impuls- und Wärmestromdichten pro Dreieck in HDF5, sodass sich Kraft- und Wärmeverteilungen später visualisieren lassen.

## 15. Definition of Done

Der Solver gilt für mich als wissenschaftlich nutzbare erste Vollversion, wenn folgende Anforderungen nachweislich erfüllt sind:

- [ ] Ein gemeinsamer C++20-Quellcode kompiliert für CPU, NVIDIA CUDA und AMD HIP.
- [ ] TPMC und DSMC verwenden dieselbe Geometrie-, GSI- und Tally-Engine.
- [ ] Maxwell und CLL sind analytisch und statistisch validiert.
- [ ] Kraft, Moment, C_D, C_L, C_M und Wärmeübertrag enthalten Einheiten, Bezugsgrößen und Vorzeichen.
- [ ] Alle Hauptgrößen enthalten statistische Fehler und Konfidenzintervalle.
- [ ] DSMC erhält bei elastischen Kollisionen Impuls und Energie innerhalb der Toleranzen.
- [ ] Numerische und statistische Konvergenz sind getrennt dokumentiert.
- [ ] CPU-, CUDA- und HIP-Ergebnisse stimmen innerhalb begründeter Toleranzen überein.
- [ ] Die festgelegten SPARTA-Vergleichsfälle bestehen.
- [ ] Jeder Lauf archiviert Konfiguration, Seed, Git-Commit, Build- und Backenddaten.
- [ ] Beispiele und automatische Unit-, Physik-, Regressions- und Performance-Tests sind vorhanden.
- [ ] Installations-, Benutzer-, Methoden- und Validierungsdokumentation ist vollständig.

## 16. Empfehlung für die tatsächliche Umsetzung

Die wichtigste technische Entscheidung ist bereits getroffen: Kokkos als gemeinsame GPU-Abstraktion mit zwei separat konfigurierten Builds.

Die Implementierungsreihenfolge würde ich strikt auf M0 bis M5 aufbauen. Insbesondere würde ich die Erweiterung zu DSMC erst angehen, sobald der TPMC-Modus einschließlich Maxwell, CLL, Mehrfachreflexionen und Energie-/Impulsbilanz verifiziert ist.

Damit vermeiden wir, gleichzeitig Fehler in der Oberflächenphysik, Partikelbewegung und intermolekularen Kollisionsmodellierung suchen zu müssen.

Das erste wirklich entscheidende Zwischenprodukt wäre deshalb ein validierter GPU-TPMC-Solver, der auf NVIDIA und AMD die vollständigen sechs aerodynamischen Ergebnisgrößen berechnet. Darauf können wir den DSMC-Modus aufbauen, ohne die bestehende Oberflächen- oder Ergebnisarchitektur neu entwickeln zu müssen.
