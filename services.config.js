// Frontend Service Configuration
// This file tells your frontend which service to call for each endpoint

const SERVICES_CONFIG = {
  // Data + WebSocket Service (Free Tier - 512 MB)
  dataWebsocket: {
    baseUrl: process.env.REACT_APP_DATA_WEBSOCKET_URL || 'https://your-data-websocket-service.onrender.com',
    endpoints: {
      // Stock Data
      stockHistory: '/stock/{symbol}/history',
      stockInfo: '/stock/{symbol}/info',
      marketStatus: '/market/status',
      tokenMapping: '/mapping/symbol-to-token',
      
      // WebSocket
      websocketStream: '/ws/stream',
      websocketHealth: '/ws/health',
      websocketConnections: '/ws/connections',
      
      // Authentication
      createToken: '/auth/token',
      verifyToken: '/auth/verify',
      
      // Health
      health: '/health'
    }
  },
  
  // Analysis Service (Standard Tier - 2 GB)
  analysis: {
    baseUrl: process.env.REACT_APP_ANALYSIS_URL || 'https://your-analysis-service.onrender.com',
    endpoints: {
      // Stock Analysis
      analyze: '/analyze',
      enhancedAnalyze: '/analyze/enhanced',
      enhancedMtf: '/analyze/enhanced-mtf',
      
      // Technical Indicators
      indicators: '/stock/{symbol}/indicators',
      patterns: '/patterns/{symbol}',
      charts: '/charts/{symbol}',
      
      // Sector Analysis
      sectorList: '/sector/list',
      sectorStocks: '/sector/{sector}/stocks',
      sectorPerformance: '/sector/{sector}/performance',
      sectorBenchmark: '/sector/benchmark',
      sectorCompare: '/sector/compare',
      stockSector: '/stock/{symbol}/sector',
      
      // User Analysis History
      userAnalyses: '/analyses/user/{user_id}',
      analysisById: '/analyses/{analysis_id}',
      analysesBySignal: '/analyses/signal/{signal}',
      analysesBySector: '/analyses/sector/{sector}',
      highConfidenceAnalyses: '/analyses/confidence/{min_confidence}',
      userAnalysisSummary: '/analyses/summary/user/{user_id}',
      
      // ML Endpoints
      mlTrain: '/ml/train',
      mlModel: '/ml/model',
      mlPredict: '/ml/predict',
      
      // Chart Management
      chartStorageStats: '/charts/storage/stats',
      cleanupCharts: '/charts/cleanup',
      cleanupSpecificCharts: '/charts/{symbol}/{interval}',
      cleanupAllCharts: '/charts/all',
      
      // Redis Management
      redisImageStats: '/redis/images/stats',
      cleanupRedisImages: '/redis/images/cleanup',
      redisImagesBySymbol: '/redis/images/{symbol}',
      cleanupRedisImagesBySymbol: '/redis/images/{symbol}',
      clearAllRedisImages: '/redis/images',
      
      // Health
      health: '/health'
    }
  }
};

// Helper function to build full URLs
export const buildServiceUrl = (serviceName, endpoint, params = {}) => {
  const service = SERVICES_CONFIG[serviceName];
  if (!service) {
    throw new Error(`Unknown service: ${serviceName}`);
  }
  
  let url = service.baseUrl + service.endpoints[endpoint];
  
  // Replace path parameters
  Object.keys(params).forEach(key => {
    url = url.replace(`{${key}}`, params[key]);
  });
  
  return url;
};

// Helper function to get service base URL
export const getServiceBaseUrl = (serviceName) => {
  const service = SERVICES_CONFIG[serviceName];
  if (!service) {
    throw new Error(`Unknown service: ${serviceName}`);
  }
  return service.baseUrl;
};

// Helper function to get endpoint path
export const getEndpointPath = (serviceName, endpoint) => {
  const service = SERVICES_CONFIG[serviceName];
  if (!service) {
    throw new Error(`Unknown service: ${serviceName}`);
  }
  return service.endpoints[endpoint];
};

export default SERVICES_CONFIG;
