## Test local deploy
curl -X POST http://127.0.0.1:8000/inference/predict   \
    -H "Content-Type: application/json"   \
    -d '{"ipv": 1, "ips":2, "iaa":3, "ieg":4, "no_av":5, "ida":6}'


## Test docker deploy
curl -X POST http://127.0.0.1:10000/inference/predict \
    -H "Content-Type: application/json" \
    -d '{"ipv": 1, "ips": 2, "iaa": 3, "ieg": 4, "no_av": 5, "ida": 6}'


## Test Render Cloud docker deploy
curl -X POST https://masters-thesis-school-lag-predictor.onrender.com/inference/predict \
  -H "Content-Type: application/json" \
  -d '{"ipv": 1, "ips": 2, "iaa": 3, "ieg": 4, "no_av": 5, "ida": 6}'