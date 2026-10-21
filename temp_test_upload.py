import requests
files = {
    'genomics_file': open('genomics_sample.csv', 'rb'),
    'lab_file': open('labs_sample.csv', 'rb')
}
try:
    r = requests.post('http://127.0.0.1:8000/test-upload?patient_zip=90210', files=files, timeout=60)
    print('STATUS', r.status_code)
    print(r.text)
except Exception as e:
    print('ERROR', repr(e))
finally:
    for f in files.values():
        f.close()
