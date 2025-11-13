// Emojis por tipo de enfermedad
export const diseaseEmojis = {
  // Manzana (Apple)
  'apple___apple_scab': '🍎🟤',
  'apple___black_rot': '🍎⚫',
  'apple___cedar_apple_rust': '🍎🦠',
  'apple___healthy': '🍎🌿',
  
  // Maíz (Corn/Maize)
  'corn_(maize)___common_rust_': '🌽🟤',
  'corn_(maize)___healthy': '🌽🌿',
  'corn_(maize)___northern_leaf_blight': '🌽🍄',
  
  // Papa (Potato)
  'potato___early_blight': '🥔🟤',
  'potato___healthy': '🥔🌿',
  'potato___late_blight': '🥔🍄',
  
  // Tomate (Tomato)
  'tomato___bacterial_spot': '🍅🦠',
  'tomato___early_blight': '🍅🟤',
  'tomato___healthy': '🍅🌿',
  'tomato___late_blight': '🍅🍄',
  'tomato___leaf_mold': '🍅🟢',
};

// Información detallada de enfermedades
export const diseaseInfo = {
  'apple___apple_scab': {
    scientificName: 'Venturia inaequalis',
    description: 'Manchas verde-oliva a marrón en hojas y frutos. Causa defoliación prematura y afecta la calidad de la fruta.',
    symptoms: ['Manchas circulares oscuras', 'Deformación de hojas', 'Lesiones en frutos'],
    treatment: 'Aplicar fungicidas preventivos (captan, mancozeb). Eliminar hojas caídas. Podar para mejorar circulación de aire.',
    prevention: 'Variedades resistentes, manejo sanitario, espaciamiento adecuado'
  },
  'apple___black_rot': {
    scientificName: 'Botryosphaeria obtusa',
    description: 'Pudrición negra que causa manchas foliares, cancros en ramas y pudrición de frutos. Altamente destructiva.',
    symptoms: ['Manchas púrpuras con bordes definidos', 'Frutos momificados', 'Cancros en ramas'],
    treatment: 'Fungicidas sistémicos (myclobutanil, difenoconazole). Podar y destruir tejido infectado. Aplicar en floración.',
    prevention: 'Higiene del huerto, poda sanitaria, eliminar frutos momificados'
  },
  'apple___cedar_apple_rust': {
    scientificName: 'Gymnosporangium juniperi-virginianae',
    description: 'Roya que requiere dos hospederos (manzano y enebro). Causa manchas amarillas-naranjas en hojas.',
    symptoms: ['Manchas amarillas brillantes', 'Pústulas naranjas', 'Defoliación temprana'],
    treatment: 'Fungicidas protectores (mancozeb, ziram). Aplicar desde botón rosa hasta 4 semanas después. Eliminar enebros cercanos.',
    prevention: 'Plantar variedades resistentes, alejar de enebros'
  },
  'corn_(maize)___common_rust_': {
    scientificName: 'Puccinia sorghi',
    description: 'Roya común que forma pústulas café-rojizas en hojas. Reduce fotosíntesis y rendimiento del cultivo.',
    symptoms: ['Pústulas ovales café-rojizas', 'Dispersión en ambas caras de hojas', 'Amarillamiento prematuro'],
    treatment: 'Fungicidas foliares (triazoles, estrobilurinas). Aplicar al detectar primeros síntomas. Rotación de cultivos.',
    prevention: 'Híbridos resistentes, siembra temprana, nutrición balanceada'
  },
  'corn_(maize)___northern_leaf_blight': {
    scientificName: 'Setosphaeria turcica',
    description: 'Tizón foliar que causa lesiones elípticas grises-verdosas. Puede reducir rendimiento hasta 50% en condiciones favorables.',
    symptoms: ['Lesiones alargadas elípticas', 'Color gris-verde a marrón', 'Coalescencia de lesiones'],
    treatment: 'Fungicidas (azoxistrobina, propiconazol). Aplicar preventivamente en zonas endémicas. Manejo de residuos.',
    prevention: 'Variedades resistentes, rotación de cultivos, enterrar residuos'
  },
  'potato___early_blight': {
    scientificName: 'Alternaria solani',
    description: 'Tizón temprano que causa manchas concéntricas en hojas. Común en condiciones cálidas y húmedas.',
    symptoms: ['Manchas circulares con anillos concéntricos', 'Amarillamiento alrededor de manchas', 'Afecta hojas inferiores primero'],
    treatment: 'Fungicidas (clorotalonil, mancozeb, azoxistrobina). Aplicar cada 7-10 días. Fertilización balanceada.',
    prevention: 'Rotación de cultivos, semilla certificada, riego por goteo'
  },
  'potato___late_blight': {
    scientificName: 'Phytophthora infestans',
    description: 'Tizón tardío devastador. Causó la hambruna irlandesa. Puede destruir cultivos en días bajo condiciones favorables.',
    symptoms: ['Lesiones húmedas gris-verdosas', 'Marchitez rápida', 'Pudrición de tubérculos'],
    treatment: 'Fungicidas sistémicos (metalaxil, mandipropamid). Aplicación preventiva obligatoria. Destruir plantas infectadas.',
    prevention: 'Monitoreo constante, variedades resistentes, evitar riego por aspersión nocturno'
  },
  'tomato___bacterial_spot': {
    scientificName: 'Xanthomonas spp.',
    description: 'Mancha bacteriana que afecta hojas, tallos y frutos. Se propaga por agua y herramientas contaminadas.',
    symptoms: ['Manchas pequeñas oscuras con halo amarillo', 'Lesiones en frutos', 'Defoliación severa'],
    treatment: 'Aplicar cobre fijo o bactericidas. Eliminar plantas severamente afectadas. Desinfectar herramientas.',
    prevention: 'Semilla tratada, rotación 3 años, evitar trabajo con plantas mojadas'
  },
  'tomato___early_blight': {
    scientificName: 'Alternaria solani',
    description: 'Tizón temprano con manchas concéntricas características. Afecta hojas maduras primero.',
    symptoms: ['Manchas con anillos concéntricos ("ojo de buey")', 'Hojas inferiores afectadas primero', 'Caída prematura de hojas'],
    treatment: 'Fungicidas (mancozeb, clorotalonil, azoxistrobina). Aplicar preventivamente. Remover hojas basales.',
    prevention: 'Mulching, riego por goteo, espaciamiento adecuado, nutrición balanceada'
  },
  'tomato___late_blight': {
    scientificName: 'Phytophthora infestans',
    description: 'Tizón tardío altamente destructivo. Puede aniquilar plantaciones enteras en 7-10 días.',
    symptoms: ['Lesiones grandes irregulares gris-verdosas', 'Moho blanco en envés', 'Pudrición de frutos'],
    treatment: 'Fungicidas sistémicos urgentes (cymoxanil, metalaxil). Destruir plantas infectadas. Aplicación preventiva crítica.',
    prevention: 'Monitoreo diario, variedades resistentes, plásticos protectores, ventilación'
  },
  'tomato___leaf_mold': {
    scientificName: 'Passalora fulva',
    description: 'Moho de la hoja común en invernaderos. Prospera en alta humedad (>85%) y poca ventilación.',
    symptoms: ['Manchas amarillas en haz', 'Moho verde-oliva en envés', 'Enrollamiento de hojas'],
    treatment: 'Fungicidas (clorotalonil, mancozeb). Mejorar ventilación. Reducir humedad. Eliminar hojas afectadas.',
    prevention: 'Variedades resistentes, ventilación adecuada, control de humedad, espaciamiento'
  }
};

// Recursos externos por enfermedad
export const diseaseResources = {
  'apple___apple_scab': [
    { title: 'Wikipedia - Sarna del Manzano', url: 'https://en.wikipedia.org/wiki/Apple_scab', type: 'encyclopedia' },
    { title: 'PlantVillage - Apple Scab', url: 'https://plantvillage.psu.edu/topics/apple/infos', type: 'guide' },
    { title: 'Extension - Manejo de Sarna', url: 'https://extension.umn.edu/plant-diseases/apple-scab', type: 'extension' },
    { title: 'EPA - Fungicidas Aprobados', url: 'https://www.epa.gov/pesticides', type: 'official' }
  ],
  'apple___black_rot': [
    { title: 'Wikipedia - Pudrición Negra', url: 'https://en.wikipedia.org/wiki/Black_rot_(apple)', type: 'encyclopedia' },
    { title: 'PlantVillage - Black Rot', url: 'https://plantvillage.psu.edu/topics/apple/infos', type: 'guide' },
    { title: 'Cornell - Black Rot Management', url: 'https://www.cornell.edu/', type: 'extension' }
  ],
  'apple___cedar_apple_rust': [
    { title: 'Wikipedia - Roya del Cedro', url: 'https://en.wikipedia.org/wiki/Gymnosporangium_juniperi-virginianae', type: 'encyclopedia' },
    { title: 'PlantVillage - Cedar Apple Rust', url: 'https://plantvillage.psu.edu/topics/apple/infos', type: 'guide' },
    { title: 'Extension - Control de Roya', url: 'https://extension.umn.edu/plant-diseases/cedar-apple-rust', type: 'extension' }
  ],
  'apple___healthy': [
    { title: 'Guía de Cultivo de Manzanas', url: 'https://extension.umn.edu/fruit/apples', type: 'guide' },
    { title: 'Manejo Integrado de Plagas', url: 'https://www.epa.gov/safepestcontrol/integrated-pest-management-ipm-principles', type: 'official' }
  ],
  'corn_(maize)___common_rust_': [
    { title: 'Wikipedia - Roya Común del Maíz', url: 'https://en.wikipedia.org/wiki/Puccinia_sorghi', type: 'encyclopedia' },
    { title: 'PlantVillage - Common Rust', url: 'https://plantvillage.psu.edu/topics/corn-maize/infos', type: 'guide' },
    { title: 'Extension - Corn Rust Management', url: 'https://extension.umn.edu/corn-pest-management/rust-corn', type: 'extension' },
    { title: 'CIMMYT - Corn Diseases', url: 'https://www.cimmyt.org/', type: 'research' }
  ],
  'corn_(maize)___healthy': [
    { title: 'Guía de Cultivo de Maíz', url: 'https://extension.umn.edu/crop-production/corn', type: 'guide' },
    { title: 'USDA - Corn Production', url: 'https://www.usda.gov/', type: 'official' }
  ],
  'corn_(maize)___northern_leaf_blight': [
    { title: 'Wikipedia - Tizón Foliar del Norte', url: 'https://en.wikipedia.org/wiki/Northern_corn_leaf_blight', type: 'encyclopedia' },
    { title: 'PlantVillage - Northern Leaf Blight', url: 'https://plantvillage.psu.edu/topics/corn-maize/infos', type: 'guide' },
    { title: 'Extension - Blight Control', url: 'https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight', type: 'extension' },
    { title: 'IPM - Manejo Integrado', url: 'https://www.epa.gov/safepestcontrol/integrated-pest-management-ipm-principles', type: 'official' }
  ],
  'potato___early_blight': [
    { title: 'Wikipedia - Alternaria (Tizón Temprano)', url: 'https://en.wikipedia.org/wiki/Alternaria_solani', type: 'encyclopedia' },
    { title: 'PlantVillage - Early Blight', url: 'https://plantvillage.psu.edu/topics/potato/infos', type: 'guide' },
    { title: 'Extension - Early Blight Management', url: 'https://extension.umn.edu/diseases/early-blight-potato-and-tomato', type: 'extension' },
    { title: 'CIP - International Potato Center', url: 'https://cipotato.org/', type: 'research' }
  ],
  'potato___healthy': [
    { title: 'Guía de Cultivo de Papa', url: 'https://extension.umn.edu/vegetables/growing-potatoes', type: 'guide' },
    { title: 'CIP - Potato Resources', url: 'https://cipotato.org/', type: 'research' }
  ],
  'potato___late_blight': [
    { title: 'Wikipedia - Phytophthora infestans', url: 'https://en.wikipedia.org/wiki/Phytophthora_infestans', type: 'encyclopedia' },
    { title: 'PlantVillage - Late Blight', url: 'https://plantvillage.psu.edu/topics/potato/infos', type: 'guide' },
    { title: 'Extension - Late Blight Management', url: 'https://extension.umn.edu/diseases/late-blight', type: 'extension' },
    { title: 'CIP - Late Blight Resources', url: 'https://cipotato.org/crops/potato/potato-diseases/late-blight/', type: 'research' },
    { title: 'USAblight - Alerta Temprana', url: 'https://usablight.org/', type: 'tool' }
  ],
  'tomato___bacterial_spot': [
    { title: 'Wikipedia - Mancha Bacteriana', url: 'https://en.wikipedia.org/wiki/Bacterial_leaf_spot', type: 'encyclopedia' },
    { title: 'PlantVillage - Bacterial Spot', url: 'https://plantvillage.psu.edu/topics/tomato/infos', type: 'guide' },
    { title: 'Extension - Bacterial Disease Control', url: 'https://extension.umn.edu/diseases/bacterial-diseases-tomato', type: 'extension' }
  ],
  'tomato___early_blight': [
    { title: 'Wikipedia - Alternaria (Tizón Temprano)', url: 'https://en.wikipedia.org/wiki/Alternaria_solani', type: 'encyclopedia' },
    { title: 'PlantVillage - Early Blight', url: 'https://plantvillage.psu.edu/topics/tomato/infos', type: 'guide' },
    { title: 'Extension - Early Blight in Tomatoes', url: 'https://extension.umn.edu/diseases/early-blight-potato-and-tomato', type: 'extension' }
  ],
  'tomato___healthy': [
    { title: 'Guía de Cultivo de Tomates', url: 'https://extension.umn.edu/vegetables/growing-tomatoes', type: 'guide' },
    { title: 'USDA - Tomato Production', url: 'https://www.usda.gov/', type: 'official' }
  ],
  'tomato___late_blight': [
    { title: 'Wikipedia - Phytophthora infestans', url: 'https://en.wikipedia.org/wiki/Phytophthora_infestans', type: 'encyclopedia' },
    { title: 'PlantVillage - Late Blight', url: 'https://plantvillage.psu.edu/topics/tomato/infos', type: 'guide' },
    { title: 'Extension - Late Blight in Tomatoes', url: 'https://extension.umn.edu/diseases/late-blight', type: 'extension' },
    { title: 'USAblight - Monitoring System', url: 'https://usablight.org/', type: 'tool' }
  ],
  'tomato___leaf_mold': [
    { title: 'Wikipedia - Moho de la Hoja', url: 'https://en.wikipedia.org/wiki/Cladosporium_fulvum', type: 'encyclopedia' },
    { title: 'PlantVillage - Leaf Mold', url: 'https://plantvillage.psu.edu/topics/tomato/infos', type: 'guide' },
    { title: 'Extension - Leaf Mold Management', url: 'https://extension.umn.edu/diseases/leaf-mold-tomato', type: 'extension' }
  ]
};

// Recursos generales
export const generalResources = [
  { title: 'PlantVillage - Base de Conocimiento', url: 'https://plantvillage.psu.edu/', type: 'general' },
  { title: 'Dataset Kaggle - Plant Disease', url: 'https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset', type: 'data' }
];
