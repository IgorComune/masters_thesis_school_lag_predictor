curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "ipv": 5,
  "ips": 6,
  "iaa": 7,
  "ieg": 4,
  "no_av": 3,
  "ida": 6
}'
