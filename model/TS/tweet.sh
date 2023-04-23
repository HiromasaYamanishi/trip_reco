export ACCESS_TOKEN='AAAAAAAAAAAAAAAAAAAAAIUwiQEAAAAAL0Zwsr4bEVMZSF7Za%2Fi%2B0F8ltvs%3DCxyufy2VQY0Y6Vg8BkH5VIfBWdIiyaq1SOGFHladSbx0OqgF6r' && \
curl -X POST 'https://api.twitter.com/2/tweets/search/stream/rules' \
-H "Content-type: application/json" \
-H "Authorization: Bearer $ACCESS_TOKEN" -d \
'{
  "add": [
    {"value": "cat has:media", "tag": "cats with media"},
    {"value": "cat has:media -grumpy", "tag": "happy cats with media"},
    {"value": "meme", "tag": "funny things"},
    {"value": "meme has:images"},
    {"value": "bounding_box:[-105.301758 39.964069 -105.178505 40.09455]"}
  ]
}'