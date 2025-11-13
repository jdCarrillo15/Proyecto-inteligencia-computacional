# Frontend - Sistema de Diagnóstico Fitopatológico

Interfaz web desarrollada en React para el sistema de clasificación de enfermedades en plantas.

## Descripción

Componente de presentación del sistema que permite a los usuarios cargar imágenes de tejido foliar, visualizar resultados de análisis y acceder a información técnica sobre patologías identificadas.

## Funcionalidades principales

- Sistema de carga mediante arrastrar y soltar
- Previsualización con capacidad de zoom (optimizado para dispositivos móviles)
- Alternancia entre modo claro y oscuro
- Indicadores visuales de estado fitosanitario
- Fichas técnicas completas (sintomatología, etiología, tratamientos)
- Módulo comparativo de tejido sano vs afectado
- Guía de buenas prácticas para captura fotográfica
- Diseño adaptable a múltiples dispositivos
- Navegación accesible mediante teclado y tecnologías de asistencia

## Requisitos del sistema

- Node.js versión 14 o superior
- Servicio backend activo en puerto 5000

## Proceso de instalación

```bash
npm install
```

## Ejecución

Verificar que el servidor backend esté en ejecución, posteriormente:

```bash
npm start
```

La aplicación iniciará automáticamente en http://localhost:3000

## Comandos disponibles

- `npm start` - Entorno de desarrollo
- `npm run build` - Compilación para producción
- `npm test` - Ejecución de pruebas
- `npm run eject` - Exposición de configuración (proceso irreversible)

## Configuración de comunicación con backend

Por defecto, la aplicación establece conexión con `http://localhost:5000`. Para modificar la configuración de puerto, ajustar el parámetro `API_URL` en `src/App.js`.

## Stack tecnológico

- React 19 - Biblioteca principal
- Axios - Cliente HTTP para peticiones asíncronas
- CSS3 - Hojas de estilo sin dependencias de frameworks
- Create React App - Configuración base del proyecto

## Módulos funcionales

**Sistema de carga:**
- Función arrastrar y soltar
- Exploración mediante selección de archivos
- Previsualización de imagen
- Formatos soportados: JPG, JPEG, PNG

**Panel de resultados:**
- Visualización de clase predicha con iconografía
- Nivel de confianza con codificación cromática
- Distribución completa de probabilidades
- Clasificación por severidad patológica

**Información técnica:**
- Nomenclatura científica de la patología
- Sintomatología característica
- Etiología común
- Protocolos de tratamiento
- Referencias bibliográficas externas

## Estructura

```
frontend/
├── public/          # Archivos estáticos
├── src/
│   ├── App.js      # Todo el código React
│   ├── App.css     # Todos los estilos
│   └── index.js    # Entry point
└── package.json
```

## Configuración

Para cambiar el puerto del backend, edita `API_URL` en `App.js`:

```javascript
const API_URL = 'http://localhost:PUERTO';
```

## Accesibilidad

Implementado siguiendo WCAG 2.1 AA:
- Navegación por teclado completa
- Etiquetas ARIA
- Contraste de colores apropiado
- Compatible con lectores de pantalla

## Proyecto Académico

Parte del curso de Inteligencia Computacional - UPTC
