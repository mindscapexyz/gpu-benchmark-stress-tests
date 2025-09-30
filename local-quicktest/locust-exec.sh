# UI mode
locust -f local-quicktest/locustfile.py --host http://127.0.0.1:8000

# # Headless mode
# locust -f locustfile.py --host http://127.0.0.1:8000 \
#   --headless -u 32 -r 4 -t 2m --csv phi3_test
