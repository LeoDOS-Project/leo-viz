//! Hand-reviewed assignments for the bundled NASA illustrations, not a flight-hardware inventory.

pub(crate) struct PartInfo {
    pub title: &'static str,
    pub description: &'static str,
    pub review: &'static str,
    pub source: &'static str,
}

pub(super) struct Spec {
    pub info: PartInfo,
    pub meshes: &'static [&'static str],
    pub materials: &'static [&'static str],
}

pub(super) struct Region {
    pub info: PartInfo,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

const PLATFORM: &str = "https://sentinels.copernicus.eu/missions/sentinel-6/satellite-description";
const PAYLOAD: &str = "https://sentinels.copernicus.eu/missions/sentinel-6/instrument-payload";
const LAYOUT: &str = "https://eo4society.esa.int/training_uploads/seasar2023/SaSAR23Theme8_FutureMissions-compendium-v2.0.pdf#page=68";
const AQUA_DOC: &str =
    "https://science.nasa.gov/mission/aqua/about-aqua-spacecraft-and-instruments/";

const OCO2_DOC: &str = "https://ocov2.jpl.nasa.gov/observatory/spacecraft/";
const AURA_DOC: &str = "https://science.nasa.gov/mission/aura/spacecraft-and-instruments/";
const GPM_DOC: &str = "https://gpm.nasa.gov/sites/default/files/imce/GPM-Press-Kit_2014.pdf";
const SNPP_DOC: &str = "https://science.nasa.gov/mission/suomi-npp/";

pub(super) const OCO2: &[Spec] = &[
    Spec {
        info: PartInfo { title: "Solar wings", description: "Two deployable wings convert sunlight into electricity for the spacecraft and its spectrometers. Batteries supply power during the night side of each orbit.", review: "The two wings, backs and braces were checked together in opposite views. Parasol in the artist's material names is not the identity of an OCO-2 instrument.", source: OCO2_DOC },
        meshes: &["Layer_0", "Layer_0 / 2", "Layer_0 / 3", "Layer_0 / 4", "Layer_0 / 5", "Layer_0 / 6", "Layer_0 / 7", "Layer_0 / 8", "Layer_0 / 9", "Layer_0 / 10"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Launch adapter", description: "The circular base is the structural interface used to attach the observatory to its launch vehicle before separation.", review: "Visual identification of the circular lower interface. The artist's SEA-Bottom Disk label is not treated as flight-hardware documentation.", source: OCO2_DOC },
        meshes: &["Layer_0 / 11"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Thermal blankets", description: "Multilayer insulation limits heat exchange with space, helping keep the optical instrument and spacecraft equipment within their operating temperatures.", review: "Gold exterior insulation grouped by visible finish and placement. These layers do not reveal the equipment beneath them.", source: OCO2_DOC },
        meshes: &[], materials: &["SEA-SeaWiffsFoil", "SEA-SidesGoldFoil", "SEA-SidesGoldFoil-sm"],
    },
    Spec {
        info: PartInfo { title: "Radiator panels", description: "Exterior thermal-control surfaces release heat to space. Keeping temperatures stable is especially important for precise spectroscopy.", review: "Visual identification of the white side-panel surfaces. No individual spectrometer channel or internal cooler is inferred from the artist's generic material names.", source: "https://ocov2.jpl.nasa.gov/observatory/instrument/" },
        meshes: &[], materials: &["SEA-SidesWhBlock"],
    },
];

pub(super) const OCO2_OTHER: PartInfo = PartInfo {
    title: "Structure and optical housing",
    description: "The main structure supports the telescope and three spectrometers, together with power, pointing and communications equipment. The spectrometers measure oxygen and carbon-dioxide absorption in reflected sunlight.",
    review: "The illustration mixes body panels, optical-housing details and array hinges within shared finishes. Its reused SeaWiFS material names are not a vetted instrument inventory. Individual internal spectrometers are not separable.",
    source: "https://ocov2.jpl.nasa.gov/observatory/instrument/",
};

pub(super) const AURA: &[Spec] = &[
    Spec {
        info: PartInfo { title: "Solar array", description: "The single deployed wing provides electrical power for the atmospheric instruments and spacecraft systems.", review: "The complete Layer_0 assembly contains panel faces, backs, braces and the root attachment. Its reused Aqua and GPM material labels are not additional spacecraft or instruments.", source: AURA_DOC },
        meshes: &["Layer_0", "Layer_0 / *"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Microwave Limb Sounder (MLS)", description: "MLS observes naturally emitted microwave radiation along the edge of Earth's atmosphere to measure trace gases, temperature and the chemistry of the ozone layer.", review: "The source-labelled MLS reflector, housing and radiator were checked against NASA's instrument layout. This is the illustrated external assembly, not a complete electronics disassembly.", source: AURA_DOC },
        meshes: &[], materials: &["aura-mls-bot", "aura-mls-radiators-new", "aura-mls-sp-ztop", "aura-mls-sp-ztop-edge", "aura-mls-top-disk-fl", "aura-mls-top-disk-sm", "MLS Mirrors"],
    },
    Spec {
        info: PartInfo { title: "HIRDLS (retired instrument)", description: "The High Resolution Dynamics Limb Sounder measured infrared emission to study atmospheric temperature, trace gases and fine vertical structure. It stopped collecting science data in 2008; its housing remains on the satellite.", review: "The explicitly labelled HIRDLS housing, aperture cover and radiator layers agree with the original instrument configuration. Retained geometry does not imply the instrument still operates.", source: AURA_DOC },
        meshes: &[], materials: &["aura-hrdls- White2", "aura-hrdls-base", "aura-hrdls-door-sm", "aura-hrdls-gengray", "aura-hrdls-goldcyl", "aura-hrdls-inst-door", "aura-hrdls-inst-inside-white", "aura-hrdls-inst-y", "aura-hrdls-inst-y-pnl", "aura-hrdls-inst-z", "aura-hrdls-inst-zwhite", "aura-hrdls-radiator", "aura-hrdls-x1", "aura-hrdls-x2", "aura-hrdls-x3panel", "aura-hrdls-y1", "aura-hrdls-y2", "aura-hrdls-z-black", "HIRDLS Gold Foil", "HIRDLS White SG"],
    },
    Spec {
        info: PartInfo { title: "Tropospheric Emission Spectrometer (retired)", description: "TES measured infrared spectra to study ozone and other gases in the lower atmosphere. Its science operations ended in 2018; the model shows the original instrument housing and optical details.", review: "The explicitly labelled TES optical and housing finishes were reviewed as one instrument. The material divisions are not separate detectors or independently verified internal mechanisms.", source: AURA_DOC },
        meshes: &[], materials: &["TES interior", "TES mirror", "TES White2", "TES White3sm", "TES White4sm", "TES white", "TESblack", "TESdetails", "TESdetailsFl", "TESGold-Y", "TESGold-Z", "TESglass", "TESgoldgen", "TESgrey", "TEStop-hood-z", "TEStop-z"],
    },
    Spec {
        info: PartInfo { title: "Communications antenna", description: "The external radio antenna supports communication between the observatory and its ground infrastructure. The radio electronics are inside the spacecraft.", review: "The source-labelled dish was visually checked. Its mesh alone does not establish a particular radio frequency or link capacity.", source: AURA_DOC },
        meshes: &[], materials: &["antenna"],
    },
    Spec {
        info: PartInfo { title: "Thrusters", description: "Small rocket thrusters provide forces for orbit maintenance and spacecraft control. They are distinct from the scientific instruments and do not produce electrical power.", review: "The three explicitly labelled thruster finishes form the illustrated nozzle assembly. Tank, plumbing and propulsion performance are not resolved by these surfaces.", source: AURA_DOC },
        meshes: &[], materials: &["aura-comm-thruster-bl", "aura-comm-thruster-gold-fl", "aura-comm-thruster-gold-sm"],
    },
    Spec {
        info: PartInfo { title: "Radiator panels", description: "Radiators reject waste heat from the spacecraft and its instruments to keep equipment within its operating temperatures.", review: "The explicitly labelled main and rear radiator panels are grouped. Instrument-specific radiators remain with their respective instruments.", source: AURA_DOC },
        meshes: &[], materials: &["aura-mainRadiators55", "aura-rear-radiators-X", "aura-rear-radiators-Y"],
    },
];

pub(super) const AURA_OTHER: PartInfo = PartInfo {
    title: "Platform and remaining payload fittings",
    description: "The spacecraft platform carries the atmospheric instruments and provides structural support, thermal protection, pointing, electrical power and data handling. OMI measures ultraviolet and visible sunlight to study ozone and air pollution.",
    review: "Remaining body layers mix panels, mounting hardware and payload details. OMI is carried on Aura but is not independently assigned from these ambiguous mesh partitions. The model is an original-configuration illustration, not a current equipment-status diagram.",
    source: AURA_DOC,
};

pub(super) const GPM: &[Spec] = &[
    Spec {
        info: PartInfo { title: "Solar arrays", description: "Two deployable solar wings generate electricity and charge the spacecraft battery. Each wing consists of four panels.", review: "Both wings' cell faces and backs were checked in opposite views. Their differently angled deployment in this illustration is retained from the source.", source: GPM_DOC },
        meshes: &[], materials: &["Solar-1", "Solar-2", "Solar-3", "Solar-4", "Solar-5", "Solar-6", "Solar-7", "Solar-8", "SolarPanelBack", "SolarPanelWhiteBack1-1", "SolarPanelWhiteBack1-2", "SolarPanelWhiteBack1-3", "SolarPanelWhiteBack1-4", "SolarPanelWhiteBack2-1", "SolarPanelWhiteBack2-2", "SolarPanelWhiteBack2-3", "SolarPanelWhiteBack2-4"],
    },
];

pub(super) const GPM_REGIONS: &[Region] = &[
    Region {
        info: PartInfo { title: "High-gain communications antenna", description: "This smaller dish transmits GMI and radar observations through NASA's Tracking and Data Relay Satellite System. It is a communications antenna, not the larger scanning microwave-imager antenna.", review: "Visual identification of the small dish at the top of the mast. Only whole connected upper surfaces are assigned; a shared mast may remain with platform fittings.", source: GPM_DOC },
        min: [-0.36,0.335,-0.41], max: [0.36,0.51,-0.03],
    },
    Region {
        info: PartInfo { title: "GMI microwave imager", description: "The large reflector collects microwave radiation emitted naturally by Earth to measure rain, snow and atmospheric moisture. The assembly rotates to produce a conical scan. Unlike the precipitation radar, GMI is a passive instrument.", review: "The elevated reflector, receiving assembly and supports are separated by whole connected surfaces, checked against NASA's GMI illustration. Even the spinningdish-top source primitive contains unrelated flat panels and is not assigned wholesale. The smaller, higher dish is communications hardware.", source: GPM_DOC },
        min: [0.055,-0.03,-0.41], max: [0.355,0.335,-0.03],
    },
    Region {
        info: PartInfo { title: "Dual-frequency precipitation radar", description: "Ku- and Ka-band radar antennas send microwave pulses into clouds and measure the returning echoes. Together they reveal the three-dimensional structure of precipitation and help distinguish rain and snow.", review: "Visual identification of the Earth-facing flat aperture panels and their housings against NASA's DPR illustration. Whole connected surfaces below the instrument deck are separated from mixed source finishes; individual Ku/Ka channel assignments are not asserted.", source: GPM_DOC },
        min: [-0.35,-0.31,-0.41], max: [0.355,0.09,-0.03],
    },
];

pub(super) const GPM_OTHER: PartInfo = PartInfo {
    title: "Platform and remaining fittings",
    description: "The observatory carries Ku- and Ka-band precipitation radars that send pulses into clouds and measure their echoes. A separate high-gain antenna sends science data through NASA's relay-satellite network. The platform provides power, pointing, propulsion and data storage.",
    review: "Several source primitives combine flat radar aperture surfaces, body panels and fittings by finish. Remaining mixed geometry is retained together rather than giving one instrument's identity to a complete material layer. Internal radar electronics are not separately modelled.",
    source: GPM_DOC,
};

pub(super) const SNPP: &[Spec] = &[
    Spec {
        info: PartInfo { title: "Solar array", description: "The deployed solar wing converts sunlight into electrical power for the spacecraft and its weather and climate instruments.", review: "The cell face is explicitly segmented in shiny_panel. The back and perimeter are separated as whole connected surfaces from the mixed tex_01 layer.", source: SNPP_DOC },
        meshes: &["shiny_panel"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Thermal blankets", description: "Gold-coloured multilayer insulation reduces unwanted heat exchange, protecting the spacecraft and instrument housings from large temperature changes.", review: "These source surfaces represent external insulation, not separate gold instruments or a view of the equipment beneath them.", source: SNPP_DOC },
        meshes: &["foil_gold"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Thermal control surfaces", description: "The silver-covered assembly provides thermal protection around the payload. Thermal control keeps sensitive optical, infrared and microwave equipment within its operating range.", review: "The foil_silver mesh is identified by finish and exterior location only. It is not asserted to be a particular instrument or the complete thermal-control subsystem.", source: SNPP_DOC },
        meshes: &["foil_silver"], materials: &[],
    },
];

pub(super) const SNPP_REGIONS: &[Region] = &[
    Region {
        info: PartInfo { title: "Solar-array back and frame", description: "The panel structure supports the solar cells, with hinges and a root attachment connecting the deployed wing to the spacecraft.", review: "Only connected surfaces outside the bus, along the deployed wing, are separated from tex_01. Body textures stay with the payload deck.", source: SNPP_DOC },
        min: [-0.24,-0.34,-0.60], max: [0.24,-0.20,0.93],
    },
];

pub(super) const SNPP_OTHER: PartInfo = PartInfo {
    title: "Payload deck and spacecraft fittings",
    description: "The deck carries VIIRS for visible and infrared imaging, infrared and microwave sounders, ozone sensors and an Earth-radiation-budget instrument. The spacecraft provides pointing, power, data storage and communications.",
    review: "This low-polygon illustration encodes many instrument details in shared textures. It does not supply independently separable geometry for each of the five original instruments, so their boundaries are not invented from tex_01, tex_02 or AO_01. Original payload geometry is retained even where an instrument has since retired.",
    source: SNPP_DOC,
};

// Each material primitive was inspected from opposite sides. In particular, do not
// group by an acronym prefix: AMR / 3 contains scattered details and a large ring.
pub(super) const SENTINEL6: &[Spec] = &[
    Spec {
        info: PartInfo { title: "Spacecraft bus", description: "The structural platform supports the payload, power, communications and attitude-control equipment.", review: "Body and exterior panels grouped across their aluminium, gold-foil and silver finishes. Internal equipment is not individually modelled.", source: PLATFORM },
        meshes: &["BUS", "BUS / 2", "BUS / 3"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Solar array", description: "The roof-shaped array supplies electrical power while the satellite's Sun angle changes along its orbit.", review: "Solar-cell faces, panel backs, trim and array attachment details are one assembly, not separate instruments.", source: PLATFORM },
        meshes: &["array", "array / 2", "array / 3", "array_locks", "array_locks / 2", "array_locks / 3"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Poseidon-4 antenna", description: "The Earth-facing radar altimeter antenna supports measurements of sea-surface height, wave height and ocean wind speed.", review: "Reflector face and foil-covered back grouped together. This is the external antenna, not the complete altimeter electronics.", source: PAYLOAD },
        meshes: &["pos4Antenna", "pos4Antenna / 2"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "AMR-C radiometer", description: "Microwave observations estimate the atmospheric water-vapour delay affecting the radar's ocean-range measurement.", review: "Reflector, housing and nearby calibration hardware are grouped. The unrelated-looking AMR / 3 primitive is deliberately excluded.", source: PAYLOAD },
        meshes: &["AMR", "AMR / 2", "AMR / 4", "AMR / 5", "AMR / 6", "AMR / 7"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "GNSS radio-occultation antennas", description: "Forward- and aft-looking arrays observe navigation signals through the atmosphere to retrieve vertical atmospheric profiles.", review: "The source calls this mesh GNSS; its separated rectangular arrays and placement match the radio-occultation layout, not the roof-mounted POD antennas.", source: PLATFORM },
        meshes: &["GNSS"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "GNSS orbit-determination antennas", description: "Roof-mounted navigation antennas support precise determination of the satellite's orbit.", review: "The source groups the roof antennas in GNSS_POD_RO_SPA. This includes the dedicated GNSS-RO POD antenna; individual receivers are not separable.", source: PLATFORM },
        meshes: &["GNSS_POD_RO_SPA"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Laser retroreflector array", description: "Reflects laser pulses from ground stations for independent, precise satellite ranging.", review: "Reflectors and their mounting housing form one assembly. Identification agrees with the source label and the Earth-facing payload layout.", source: PAYLOAD },
        meshes: &["LRA", "LRA / 2", "LRA / 3"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "DORIS antenna", description: "Receives ground-beacon signals used for precise orbit determination.", review: "Source-labelled DORIS-A antenna, checked against the Earth-facing instrument layout.", source: PAYLOAD },
        meshes: &["DORIS-A"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Star trackers", description: "Three optical heads determine spacecraft attitude from star observations.", review: "The three heads are combined in the source mesh. They are an attitude-sensing assembly, not cameras for observing Earth.", source: PLATFORM },
        meshes: &["starTrackers"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Coarse Earth and Sun sensors", description: "Coarse direction measurements support spacecraft attitude acquisition and safe-mode orientation.", review: "The source distributes several small CESS heads around the body. Their metal, housing and sensing faces are grouped rather than listed as three parts.", source: LAYOUT },
        meshes: &["CESS", "CESS / 2", "CESS / 3"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "S-band antenna", description: "Supports the bidirectional command and spacecraft-control radio link.", review: "Visual identification of the source SHA mesh as the nadir communications antenna, checked against ESA's labelled layout. The source acronym is not treated as a specification.", source: LAYOUT },
        meshes: &["SHA"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "X-band antenna", description: "Downlinks stored science observations to ground stations.", review: "The source-labelled XBA mesh matches the Earth-facing communications antenna location. Only the external antenna is represented.", source: PLATFORM },
        meshes: &["XBA"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Launch adapter ring", description: "The rear structural interface connects the spacecraft to its launch adapter before separation.", review: "Visual identification from the circular rear interface and ESA's labelled layout. MPPS is the artist's group name; no expansion of that acronym is asserted.", source: LAYOUT },
        meshes: &["MPPS", "MPPS / 2"], materials: &[],
    },
];

// Aqua is largely partitioned by finish, not by instrument. Do not infer AIRS,
// MODIS or CERES boundaries from Layer numbers or assign a mixed layer to one instrument.
pub(super) const AQUA: &[Spec] = &[
    Spec {
        info: PartInfo { title: "Spacecraft bus and thermal blankets", description: "The main structure supports the payload and contains power, radio and control electronics. Its gold-coloured multilayer blankets reduce unwanted heating and cooling; they are insulation, not solar cells.", review: "The long deck and aft body blankets are identified by their geometry. Internal equipment is not individually modelled.", source: AQUA_DOC },
        meshes: &["Aqua_Reduced_v0014 [Aqua_Layered_02 [(unnamed)]]", "Aqua_Reduced_v0014 [Aqua_Layered_02 [Layer22]]", "Aqua_Reduced_v0014 [Aqua_Layered_02 [Layer23]]"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Solar array", description: "The deployed solar wing supplies power to Aqua's spacecraft and instruments.", review: "Panel backs, cell faces, frames and support rails were visually checked and grouped across five source material layers.", source: AQUA_DOC },
        meshes: &[], materials: &["SolarPanelBack", "Gold_Metal", "SolarPanelFaces", "Solar_Cells-1", "Solar_Cells-2"],
    },
    Spec {
        info: PartInfo { title: "AMSR-E reflector assembly", description: "The large microwave reflector collects radiation used to study water, ice, precipitation and ocean conditions.", review: "Visual identification of the large reflector and its supporting structure against NASA's Aqua layout. Generic antenna material names alone do not establish the instrument identity.", source: "https://ntrs.nasa.gov/api/citations/20040171846/downloads/20040171846.pdf#page=2" },
        meshes: &[], materials: &["antenna_base", "antenna_dish_inner", "antenna_dish_outer", "antenna_poles", "antenna_poles_joints"],
    },
    Spec {
        info: PartInfo { title: "Instrument radiator panels", description: "Radiator surfaces release heat from instrument equipment to space.", review: "The artist labels these surfaces HSB Thermal Radiator Panels. This identifies radiator surfaces only; the complete HSB instrument is not independently segmented.", source: AQUA_DOC },
        meshes: &[], materials: &["HSB Thermal Radiator Panels"],
    },
];

pub(super) const SENTINEL_OTHER: PartInfo = PartInfo {
    title: "Other exterior details",
    description: "Unassigned exterior geometry retained from the original NASA illustration.",
    review: "AMR / 3 combines a large ring and scattered details. Its name is insufficient to associate all of it with the radiometer, so it is not assigned to a specific instrument.",
    source: PLATFORM,
};

pub(super) const AQUA_OTHER: PartInfo = PartInfo {
    title: "Deck, mounts and remaining fittings",
    description: "The instrument deck and mounting fittings hold the sensors in alignment and provide mechanical and thermal connections to the spacecraft. Small unlabelled fittings are not automatically separate scientific instruments.",
    review: "The artist's layers mix several pieces by material. Connected surfaces are separated for the reviewed educational annotations; remaining fittings are not assigned an unsupported instrument or circuit-board identity.",
    source: AQUA_DOC,
};

// Normalised model-space regions, reviewed against NASA's labelled configuration
// diagram (press-kit p. 11). Only whole connected surfaces that fit a region are
// assigned: no triangle is cut and no instrument is inferred from a material name.
const AQUA_LAYOUT: &str = "https://assets.science.nasa.gov/content/dam/science/missions/aqua/references/Aqua_Press_Kit.pdf#page=11";
pub(super) const AQUA_REGIONS: &[Region] = &[
    Region {
        info: PartInfo { title: "X-band communications dish", description: "The small dish sends scientific observations to Earth. It is carried on a long boom to keep its view of ground stations clear of the spacecraft and the aft instruments.", review: "Visual identification of the aft small dish against NASA's configuration diagram. Shared boom geometry can remain with the structural fittings; this is not the large AMSR-E microwave reflector.", source: AQUA_LAYOUT },
        min: [-0.32,0.34,0.50], max: [0.20,0.55,0.95],
    },
    Region {
        info: PartInfo { title: "CERES scan head 1", description: "CERES measures reflected sunlight and outgoing thermal radiation to study Earth's energy balance. Its scanning head points the radiometer across Earth rather than producing a conventional colour photograph.", review: "Visual identification of one of the two aft CERES heads. The numbering distinguishes the illustrated heads and does not assert an FM-3/FM-4 assignment.", source: AQUA_LAYOUT },
        min: [-0.30,0.22,0.71], max: [-0.09,0.35,0.82],
    },
    Region {
        info: PartInfo { title: "CERES scan head 2", description: "Aqua carries two CERES radiometers. Their complementary scanning observations help determine how much solar energy Earth absorbs and how much heat it emits to space.", review: "Visual identification of the second aft scan head against the paired CERES callout. The flight-model serial identity is not inferred.", source: AQUA_LAYOUT },
        min: [-0.30,0.115,0.55], max: [-0.09,0.22,0.90],
    },
    Region {
        info: PartInfo { title: "Small nozzle-like fittings", description: "These small angled fittings resemble thruster assemblies. A thruster expels propellant to change spacecraft motion, unlike an instrument aperture that receives radiation. Their exact hardware identity is not confirmed in this illustration.", review: "Separated from the CERES viewing heads by their distinct position and geometry. The nozzle interpretation is an explicit visual inference, not a manufacturer identification.", source: AQUA_LAYOUT },
        min: [-0.30,0.22,0.55], max: [-0.09,0.35,0.90],
    },
    Region {
        info: PartInfo { title: "MODIS housing and Earth shield", description: "MODIS measures reflected and emitted radiation in 36 spectral bands to observe clouds, vegetation, oceans, fires and surface temperature. Its housing and shield limit unwanted light and heat entering the optical system.", review: "Illustration-level visual identification of the forward large instrument housing and shield, next to AMSR-E, using NASA's labelled configuration. Not an engineering disassembly.", source: AQUA_LAYOUT },
        min: [-0.28,-0.41,0.56], max: [-0.045,-0.255,0.91],
    },
    Region {
        info: PartInfo { title: "AMSU-A1 sounding unit", description: "AMSU-A measures microwave emission used to retrieve atmospheric temperature and moisture. Microwave measurements complement infrared sounding because they can provide information through many cloud conditions.", review: "Visual match to the forward AMSU-A1 position in NASA's configuration diagram. Only connected exterior surfaces within the illustrated unit are grouped.", source: AQUA_LAYOUT },
        min: [-0.27,-0.255,0.56], max: [-0.04,-0.12,0.87],
    },
    Region {
        info: PartInfo { title: "AIRS optical housing and shield", description: "AIRS measures thousands of infrared spectral channels to retrieve atmospheric temperature, water vapour and trace gases. The optical aperture and shield control which radiation reaches the cooled detector.", review: "Visual match to the central AIRS housing and shield in the labelled layout. Adjacent microwave heads are separated into their own surface regions where the source geometry permits.", source: AQUA_LAYOUT },
        min: [-0.28,-0.07,0.70], max: [-0.05,0.115,0.89],
    },
    Region {
        info: PartInfo { title: "HSB-region sounding head", description: "The Humidity Sounder for Brazil measured microwave radiation sensitive to atmospheric water vapour. Its viewing head and surrounding shielding direct radiation into the receiver rather than transmitting data to Earth.", review: "Visual match to the HSB area beside the source-labelled HSB radiator panels. This illustration-level identification does not establish every internal receiver surface.", source: AQUA_LAYOUT },
        min: [-0.29,0.024,0.62], max: [-0.045,0.13,0.73],
    },
    Region {
        info: PartInfo { title: "Microwave sounding head", description: "A microwave sounding head collects naturally emitted radiation to study atmospheric temperature and moisture. Its opening and shielding give it a controlled view of Earth, distinct from the radio link used to send data home.", review: "Located in the AMSU sounding area of the labelled layout; no unsupported AMSU-A1/A2 receiver-channel assignment is made for this separate head.", source: AQUA_LAYOUT },
        min: [-0.29,-0.12,0.60], max: [-0.045,-0.035,0.73],
    },
    Region {
        info: PartInfo { title: "Small sensor apertures and baffles", description: "Small apertures admit radiation to a sensor, while nearby hoods and baffles restrict unwanted views and radiation. The exterior shape alone is not enough to determine a sensor's wavelength or electronic function.", review: "Adjacent small heads and baffles are separated from the main housings. Their detailed receiver identity is not asserted from generic artist material names.", source: AQUA_LAYOUT },
        min: [-0.29,-0.12,0.55], max: [-0.045,0.13,0.86],
    },
];

pub(super) const LANDSAT8: &[Spec] = &[
    Spec {
        info: PartInfo { title: "Solar array", description: "Photovoltaic cells turn sunlight into electrical power. The deployed wing supplies the spacecraft and charges the battery for the dark part of each orbit.", review: "Solar faces, backs, trim and cell layers were checked visually. Reused LRO material names describe the artist's texture library, not another spacecraft's flight hardware.", source: "https://www.usgs.gov/landsat-missions/landsat-8" },
        meshes: &[], materials: &["LDCM-solarpanel-back", "LDCM-solarpanel-face", "LDCM-solarpanel-red", "LRO-solarpanel-1", "LRO-solarpanel-2", "LRO-solarpanel-3", "LRO-solarpanel-4", "LRO-solarpanel-5", "LRO-solarpanel-6", "LRO-solarpanel-7", "LRO-solarpanel-bronze", "LRO-solarpanel-gold"],
    },
    Spec {
        info: PartInfo { title: "Thruster nozzles", description: "Small rocket nozzles provide controlled impulses for spacecraft manoeuvres. They maintain the required orbit rather than providing continuous propulsion.", review: "The source explicitly labels the inner and outer nozzle surfaces. This grouping is external nozzle geometry, not tanks, valves or the complete propulsion system.", source: "https://science.nasa.gov/mission/landsat-8/" },
        meshes: &[], materials: &["LDCM-thruster-in", "LDCM-thruster-out"],
    },
    Spec {
        info: PartInfo { title: "Spacecraft bus", description: "The main platform carries the instruments, recorder, radio and power equipment. The solid-state recorder stores observations until they can be sent to a ground station.", review: "Main panel and section materials identify the central bus. The ruler is calibrated to the published 3 m bus length, not an invented deployed envelope.", source: "https://www.usgs.gov/landsat-missions/landsat-8" },
        meshes: &[], materials: &["LDCM-mainpanels-fl", "LDCM-mainpanels-sm", "LDCM-mainsections"],
    },
];

pub(super) const ICESAT2: &[Spec] = &[
    Spec {
        info: PartInfo { title: "Spacecraft bus", description: "The spacecraft platform supplies power, communications and pointing control for the laser altimeter. Accurate attitude knowledge is essential to locate each laser footprint on Earth.", review: "The source-labelled BUS is the platform exterior; internal avionics and tanks are not individually segmented.", source: "https://science.nasa.gov/mission/icesat-2/" }, meshes: &["BUS"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "ATLAS telescope", description: "ATLAS times returning green laser photons to measure surface height. Its receiving telescope collects very faint echoes; this is a laser-ranging instrument, not a conventional colour camera.", review: "The five Atlas scope material primitives have been combined into the optical assembly. The complete ATLAS electronics are not isolated by these meshes.", source: "https://science.nasa.gov/mission/icesat-2/" }, meshes: &["Atlas scope", "Atlas scope / 2", "Atlas scope / 3", "Atlas scope / 4", "Atlas scope / 5"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Solar array and hinges", description: "The solar wing generates electrical power and unfolds after launch. Its hinge and support structure hold the panels in their deployed position.", review: "Solar-panel faces and backs are grouped with the inspected hinge meshes.", source: "https://icesat-2.gsfc.nasa.gov/sites/default/files/page_files/FactSheet_0828_2018_0.pdf" }, meshes: &["solar_panels", "solar_panels / 2", "hinges", "hinges / 2", "hinges / 3"], materials: &[],
    },
];

pub(super) const SMAP: &[Spec] = &[
    Spec {
        info: PartInfo { title: "Mesh reflector", description: "The 6 m reflector focuses weak L-band microwave signals into the receiver. The large collecting area allows soil-moisture measurements over a wide scanning swath.", review: "Both reflector faces and their support lattice are combined. These are duplicated opposite-face meshes, not two antennas. The display uses a simplified gold surface and perimeter truss instead of the source's dense triangular web; the downloaded geometry and textures remain unchanged.", source: "https://www.jpl.nasa.gov/news/press_kits/smaplaunch.pdf" }, meshes: &["antenna_01", "antenna_01.001", "antenna_03", "antenna_03.001"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Solar arrays", description: "Solar cells provide electrical power for the spacecraft and science instrument. Their supporting panels keep the cells oriented and mechanically supported.", review: "The array's textured fronts and opposite faces are combined.", source: "https://www.jpl.nasa.gov/news/press_kits/smaplaunch.pdf" }, meshes: &["array", "array.001"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Antenna boom and feed assembly", description: "The deployed boom holds the reflector away from the spacecraft. The antenna assembly scans Earth's surface as it rotates, rather than taking a normal camera photograph.", review: "Inspected arm and feed-support meshes are grouped; no individual motor or electronics identity is assigned from material names.", source: "https://www.jpl.nasa.gov/news/press_kits/smaplaunch.pdf" }, meshes: &["arm_etc", "arm_etc / 2", "arm_etc.001"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Spacecraft bus", description: "The bus contains power, communications, attitude control and support equipment. Thermal blankets limit unwanted heat exchange so the equipment can operate within its temperature range.", review: "The two body meshes are combined across their finishes. Internal equipment is not shown separately.", source: "https://www.jpl.nasa.gov/news/press_kits/smaplaunch.pdf" }, meshes: &["body.000", "body.003"], materials: &[],
    },
];

// These particular Layer nodes were visually checked as folded panel assemblies.
// The wildcard only includes their material primitives; it is not a naming heuristic.
pub(super) const MIRATA: &[Spec] = &[
    Spec {
        info: PartInfo { title: "Folded solar panels", description: "Photovoltaic cells provide electrical power to this small weather-sensor demonstrator. In this illustration the panel leaves lie against the sides of the CubeSat rather than showing the deployed wings.", review: "The four inspected panel nodes and their material primitives are grouped. ICECube and Afta material names are reused artist assets, not verified instrument identities.", source: "https://www.haystack.mit.edu/wp-content/uploads/2020/07/conf_NEROC2016_cahoy_web.pdf" }, meshes: &["Layer_1", "Layer_1 / *", "Layer_2", "Layer_2 / *", "Layer_3", "Layer_3 / *", "Layer_4", "Layer_4 / *"], materials: &[],
    },
];

pub(super) const LANDSAT8_OTHER: PartInfo = PartInfo { title: "Optical payload and external fittings", description: "OLI measures reflected sunlight in nine spectral bands, while TIRS measures emitted thermal radiation in two bands. Optical openings, baffles and thermal covers protect these measurements from unwanted light and heat.", review: "The remaining material primitives mix multiple instruments and external fittings. They are not assumed to form one physical instrument.", source: "https://science.nasa.gov/mission/landsat-8/" };
pub(super) const ICESAT2_OTHER: PartInfo = PartInfo { title: "Instrument deck and external fittings", description: "The optical payload also needs thermal protection, mounting interfaces and spacecraft fittings. These support accurate pointing and protect the detector from stray light and temperature changes.", review: "The source's generic instruments primitives combine multiple exterior objects; individual electronics identities are not established.", source: "https://science.nasa.gov/mission/icesat-2/" };
pub(super) const SMAP_OTHER: PartInfo = PartInfo {
    title: "Auxiliary source geometry",
    description: "Additional geometry retained from the original illustration.",
    review:
        "The source contains a transparent helper mesh. It is not identified as flight hardware.",
    source: "https://smap.jpl.nasa.gov/resources/99/smap-3d-model/",
};
pub(super) const MIRATA_OTHER: PartInfo = PartInfo { title: "CubeSat body and payload apertures", description: "The 3U body packages a microwave radiometer and a GPS radio-occultation experiment with the power, radio and control electronics. The two sensing techniques were designed to test a new method of radiometer calibration.", review: "The exterior is illustrated, but material names do not identify individual circuit boards or sensors. Body dimensions refer to the stowed 3U format.", source: "https://www.nasa.gov/wp-content/uploads/2016/07/elana_14_jpss1_cubesat_factsheet_508.pdf" };

const GOES_DOC: &str = "https://www.nesdis.noaa.gov/our-satellites/currently-flying/goes-east-west/goes-r-series-spacecraft-instruments";
const GOES_LAYOUT: &str =
    "https://ospo.noaa.gov/resources/documents/GOES-RSeriesDataBook.pdf#page=23";
pub(super) const GOESR: &[Spec] = &[
    Spec {
        info: PartInfo { title: "Spacecraft bus", description: "The central platform carries power, propulsion, computers and communications equipment. Thermal blankets and exterior panels help keep that equipment at suitable temperatures.", review: "Original BUS material primitives grouped. This is a GOES-R/S/T illustration, not a GOES-U model with CCOR-1.", source: GOES_DOC },
        meshes: &["BUS", "BUS / *"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Solar array", description: "The five-panel wing converts sunlight into electricity. Hinges allow it to fold for launch, while the supporting structure holds the deployed panels in alignment.", review: "Cell faces, frames, hinges, braces and edge fittings were checked together against the deployed layout.", source: GOES_LAYOUT },
        meshes: &["arrayPanels", "arrayPanels / *", "arrayHinges", "arrayHinges / *", "arrayBrace", "arrayBrace / *", "arraySensors"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Solar-array drive", description: "The rotating joint changes the array's orientation so it can face the Sun while the spacecraft keeps its weather instruments pointed at Earth.", review: "Source-labelled arrayGimble and its material primitives. Internal motors and drive electronics are not individually modelled.", source: GOES_LAYOUT },
        meshes: &["arrayGimble", "arrayGimble / *"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Magnetometer boom", description: "The long boom moves the magnetic sensors away from electrical equipment and magnetic materials in the spacecraft, reducing interference with their measurements.", review: "Boom, deployment housing and wiring grouped. The ruler uses the documented approximately 8.5 m deployed boom, not the stowed spacecraft envelope.", source: GOES_LAYOUT },
        meshes: &["Boom", "boomHousing", "boomHousing / *", "boomWires"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Magnetometer sensor heads", description: "Two sensor heads measure the surrounding magnetic field. Comparing their readings helps distinguish the natural field from the spacecraft's own magnetic influence.", review: "The source's boomEquip surfaces occupy the two outboard sensor positions in the labelled layout.", source: "https://www.ncei.noaa.gov/products/goes-r-magnetometer" },
        meshes: &["boomEquip", "boomEquip / *"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Solar observatories (SUVI / EXIS)", description: "SUVI images the Sun's hot outer atmosphere. EXIS monitors solar ultraviolet and X-ray output, including flares that can disturb radio links and navigation.", review: "The shared source assembly combines both instruments and their platform. It is not split into unsupported individual optical channels.", source: GOES_DOC },
        meshes: &["SUVI_EXIS", "SUVI_EXIS / *", "SUVI_EXIS.001", "GOES_R_2011_exp:polySurface279.001"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Advanced Baseline Imager (ABI)", description: "ABI scans Earth in 16 visible and infrared bands to observe clouds, storms, surface temperatures, smoke and other environmental features.", review: "The labelled ABI housing, optical opening and exterior fittings are grouped across their original finishes.", source: GOES_DOC },
        meshes: &["ABI", "ABI / *"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Geostationary Lightning Mapper (GLM)", description: "GLM detects brief optical flashes from lightning, including flashes inside clouds. Changes in lightning activity help forecasters assess developing severe storms.", review: "The source-labelled optical assembly and its adjacent cable are grouped.", source: GOES_DOC },
        meshes: &["GLM", "GLM / *", "glmCoax"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Earth-pointing instrument platform", description: "This platform supports the Earth-viewing instruments and isolates them from spacecraft vibration, helping preserve precise pointing and sharp observations.", review: "Bench and BusBenchTrans match the isolated Earth-pointing platform in the NOAA layout.", source: GOES_LAYOUT },
        meshes: &["Bench", "Bench / *", "BusBenchTrans", "BusBenchTrans / *"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Star trackers", description: "Optical heads compare observed stars with a catalogue to determine spacecraft orientation. They provide attitude knowledge, rather than pictures of Earth's surface.", review: "The two original tracker groups and their baffles are retained together; no individual flight-unit identity is inferred.", source: GOES_LAYOUT },
        meshes: &["Tracker1", "Tracker1 / *", "Trackers2", "Trackers2 / *"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Inertial measurement unit", description: "Inertial sensors measure spacecraft motion between star observations, helping the control system keep the instruments pointing steadily.", review: "Source-labelled IMu housing, colocated with the star trackers on the instrument platform.", source: GOES_LAYOUT },
        meshes: &["GOES_R_2011_exp:IMu.001", "GOES_R_2011_exp:IMu.001 / *"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Instrument radiator", description: "Radiators release waste heat to space. Stable instrument temperatures are important for sensitive optics and consistent measurements.", review: "The source-labelled radiator and its backing are one thermal assembly.", source: GOES_LAYOUT },
        meshes: &["GOES_R_2011_exp:Radiator.001", "GOES_R_2011_exp:Radiator.001 / *"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Space-environment sensor assembly", description: "SEISS monitors energetic electrons, protons and ions around the satellite. These observations support warnings about radiation that can damage spacecraft electronics.", review: "Illustration-level identification of the source SEIS assembly; individual particle-detector channels are not separately assigned.", source: GOES_DOC },
        meshes: &["SEIS", "SEIS / *"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Orbit-raising engine", description: "The main engine provides larger manoeuvres needed to reach the operational orbit. It is distinct from the smaller thrusters used for routine pointing and station keeping.", review: "The central bell-shaped source Thruster and its rim are grouped. This is not a mapping of every small attitude-control nozzle.", source: GOES_LAYOUT },
        meshes: &["Thruster", "ThrustRim"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Communications antenna wing", description: "The projecting wing gives radio antennas a clear view of Earth, reducing obstruction by the main spacecraft structure.", review: "Visual identification of Inst3 against the labelled communications wing; individual frequencies are not inferred from material names.", source: GOES_LAYOUT },
        meshes: &["Inst3", "Inst3 / *"], materials: &[],
    },
    Spec {
        info: PartInfo { title: "Side-mounted antenna assembly", description: "External radio antennas connect the satellite with ground stations, carrying instrument data, commands and spacecraft health information.", review: "The separate side-mounted Ins1 assembly is retained without an unsupported assignment of each small surface to a radio band.", source: GOES_LAYOUT },
        meshes: &["Ins1", "Ins1 / *"], materials: &[],
    },
];
pub(super) const GOESR_OTHER: PartInfo = PartInfo { title: "Remaining exterior fittings", description: "Small mounting and exterior fittings support the spacecraft equipment. Their detailed identity is not established by this illustration.", review: "Unlabelled pCube geometry is not assigned a scientific instrument solely from its position or artist name.", source: GOES_LAYOUT };

const SWOT_DOC: &str = "https://swot.jpl.nasa.gov/resources/91/swot-payload/";
const SWOT_SCIENCE: &str =
    "https://swot.jpl.nasa.gov/internal_resources/638/swot-press-kit.pdf#page=23";
pub(super) const SWOT_REGIONS: &[Region] = &[
    Region { info: PartInfo { title: "Solar wing A", description: "Solar cells turn sunlight into electrical power. The wing's panels and hinges fold for launch, then deploy and rotate to follow the Sun.", review: "Whole connected panel faces, backs and hinge fittings on the negative-X side of the source illustration.", source: "https://www.jpl.nasa.gov/press-kits/swot/quick-facts/" }, min: [-1.0,0.04,-0.072], max: [-0.075,0.183,0.072] },
    Region { info: PartInfo { title: "Solar wing B", description: "The second solar wing works with the opposite array to provide power for radar observations and the spacecraft's support systems.", review: "The positive-X wing. A/B distinguish the illustration's wings, not manufacturer flight-unit designations.", source: "https://www.jpl.nasa.gov/press-kits/swot/quick-facts/" }, min: [0.075,0.04,-0.072], max: [1.0,0.183,0.072] },
    Region { info: PartInfo { title: "KaRIn antenna A", description: "This radar antenna works with the antenna at the other end of the mast. Differences in their returning signals allow the height of water to be measured across a wide swath.", review: "The complete negative-Z reflectarray and its end fittings. A/B are illustration labels, not port/starboard flight assignments.", source: SWOT_SCIENCE }, min: [-0.255,0.015,-0.55], max: [0.255,0.060,-0.49] },
    Region { info: PartInfo { title: "KaRIn antenna B", description: "The second antenna provides the other end of the radar's approximately 10 m measurement baseline. The pair measures water height rather than producing normal camera photographs.", review: "The complete positive-Z reflectarray and end fittings.", source: SWOT_SCIENCE }, min: [-0.255,0.015,0.49], max: [0.255,0.060,0.55] },
    Region { info: PartInfo { title: "KaRIn mast and deployment joints", description: "The deployable mast holds the two radar antennas far apart. Its stable geometry is essential because motion between the antennas can affect the inferred water height.", review: "Whole connected mast segments and small hinge fittings, separated from the antenna faces without cutting triangles.", source: SWOT_SCIENCE }, min: [-0.012,0.044,-0.55], max: [0.012,0.079,0.55] },
    Region { info: PartInfo { title: "Nadir radar altimeter", description: "The Earth-facing altimeter measures the radar signal's round-trip travel time to obtain water height directly below the spacecraft, filling the gap between KaRIn's two swaths.", review: "Visual match of the larger Earth-facing dish and supports to the labelled nadir payload layout.", source: SWOT_DOC }, min: [0.0,-0.141,-0.063], max: [0.13,-0.116,0.063] },
    Region { info: PartInfo { title: "Microwave radiometer", description: "Natural microwave emissions reveal atmospheric water vapour. Those measurements correct the radar signals' atmospheric delay, improving the water-height estimates.", review: "Visual match of the smaller offset reflector, feed and mounting structure to the labelled payload diagram.", source: SWOT_DOC }, min: [-0.17,-0.205,-0.055], max: [-0.025,-0.12,0.055] },
    Region { info: PartInfo { title: "DORIS and laser-ranging fixtures", description: "DORIS receives signals from ground radio beacons, while the laser reflector returns pulses to ground ranging stations. Both help determine the spacecraft's orbit accurately.", review: "The two illustrated tracking posts are grouped. Their mixed material surfaces do not establish a reliable individual head mapping.", source: SWOT_DOC }, min: [0.057,-0.209,-0.086], max: [0.074,-0.08,0.086] },
    Region { info: PartInfo { title: "X-band downlink antenna", description: "This compact radio antenna sends stored scientific observations to Earth. It transmits data, unlike the nearby radar antennas that make the measurements.", review: "Visual identification of the small choke-ring antenna on the nadir deck using the labelled payload diagram.", source: SWOT_DOC }, min: [0.05,-0.179,-0.008], max: [0.075,-0.16,0.008] },
    Region { info: PartInfo { title: "Star trackers", description: "These optical heads recognize star patterns to determine the satellite's orientation. Accurate attitude knowledge helps place the radar measurements correctly on Earth.", review: "Paired optical heads and their baffles at the upper end of the source body.", source: "https://swot.jpl.nasa.gov/internal_resources/638/swot-press-kit.pdf#page=25" }, min: [-0.07,0.15,-0.10], max: [0.07,0.21,-0.05] },
];
pub(super) const SWOT_OTHER: PartInfo = PartInfo { title: "Spacecraft bus and thermal blankets", description: "The central structure houses the computers, power and propulsion equipment. Gold-coloured insulation and radiator surfaces regulate heat; the exterior blankets are not solar cells.", review: "Original mixed layers retained around the reviewed connected assemblies. Small unlabelled fittings are not given invented hardware identities.", source: "https://swot.jpl.nasa.gov/internal_resources/638/swot-press-kit.pdf#page=25" };

const GRACE_DOC: &str = "https://www.jpl.nasa.gov/news/press_kits/grace-fo/mission/spacecraft/";
pub(super) const GRACEFO: &[Spec] = &[
    Spec { info: PartInfo { title: "Spacecraft body and solar cells", description: "Solar cells are integrated into the top and sloping sides instead of extending on large wings. The body supports the instruments and encloses the computers, batteries and nitrogen propellant tanks.", review: "The two source BUS finishes combine body panels, insulation and solar-cell surfaces. They are not an internal engineering disassembly.", source: GRACE_DOC }, meshes: &["BUS", "BUS / 2"], materials: &[] },
];
pub(super) const GRACEFO_REGIONS: &[Region] = &[
    Region { info: PartInfo { title: "Microwave ranging horn", description: "The horn exchanges K- and Ka-band signals with the other GRACE-FO satellite. Tiny changes in their separation reveal variations in Earth's gravity field.", review: "Visual identification of the forward horn, feed and aperture surround. The internal microwave electronics are not visible.", source: GRACE_DOC }, min: [-0.08,0.025,0.78], max: [0.08,0.19,0.85] },
    Region { info: PartInfo { title: "Zenith navigation antenna", description: "The sky-facing antenna receives navigation signals used to determine the spacecraft's orbit and accurately time its scientific measurements.", review: "Visual identification of the top-mounted antenna housing; no claim is made about receiver electronics inside the bus.", source: "https://research-infrastructure.gfz.de/en/infrastructure/9" }, min: [-0.08,0.205,-0.23], max: [0.08,0.27,-0.07] },
    Region { info: PartInfo { title: "Aft radio-occultation antenna", description: "This rear-looking antenna receives navigation signals passing through Earth's atmosphere. Their bending helps retrieve atmospheric temperature and water-vapour profiles.", review: "Illustration-level identification of the rear antenna, distinct from the forward microwave ranging horn.", source: "https://research-infrastructure.gfz.de/en/infrastructure/9" }, min: [-0.06,-0.06,-0.85], max: [0.06,0.06,-0.68] },
    Region { info: PartInfo { title: "Star trackers and protective baffles", description: "Star cameras determine the satellite's precise orientation. Their hoods shield the optics from unwanted light, allowing accurate measurements without viewing Earth as a normal imaging camera would.", review: "The silver hoods and three adjacent optical heads are grouped; individual camera serial numbers are not inferred.", source: "https://www.gfz.de/en/section/global-geomonitoring-and-gravity-field/projects/grace-fo-gravity-recovery-and-climate-experiment-follow-on-mission/satellites" }, min: [-0.33,0.02,-0.11], max: [0.33,0.238,0.35] },
    Region { info: PartInfo { title: "Laser retroreflector", description: "Corner-cube reflectors send laser pulses back to ground stations. Their travel time provides an independent check of the satellite's orbit; this is separate from the laser link between the two spacecraft.", review: "Visual identification of the small underside reflector assembly. It is not labelled as the internal laser ranging interferometer.", source: GRACE_DOC }, min: [0.30,-0.19,0.73], max: [0.38,-0.15,0.80] },
    Region { info: PartInfo { title: "Forward launch mounting feet", description: "These mechanical interfaces support and attach the satellite to its launch system before separation. They are structural fittings, not downward-looking scientific sensors.", review: "The forward pair of repeated underside ring mounts, identified by their geometry and launch-support role.", source: GRACE_DOC }, min: [-0.26,-0.18,0.43], max: [0.26,-0.14,0.54] },
    Region { info: PartInfo { title: "Aft launch mounting feet", description: "The second pair of mounting interfaces distributes launch loads through the spacecraft structure. These fittings remain on the exterior after deployment.", review: "The aft pair of underside mounts, kept separate from the small central exterior fittings.", source: GRACE_DOC }, min: [-0.26,-0.18,-0.55], max: [0.26,-0.14,-0.45] },
];
pub(super) const GRACEFO_OTHER: PartInfo = PartInfo { title: "Exterior sensors and fittings", description: "Additional apertures, antennas and mounting fittings support navigation, control and communications. Internal accelerometers and laser electronics are not exposed in this exterior illustration.", review: "Remaining mixed-source surfaces are not mapped to specific electronic units without identifiable geometry.", source: GRACE_DOC };
