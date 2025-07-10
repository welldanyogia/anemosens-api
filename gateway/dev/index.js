const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(
  '/predict',
  createProxyMiddleware({
    target: 'http://predictor:5000',
    changeOrigin: true,
    pathRewrite: { '^/predict': '/predict' }
  })
);

app.listen(PORT, () => {
  console.log(`API Gateway (dev) listening on port ${PORT}`);
});