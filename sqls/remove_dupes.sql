
# Keep the original tweet only
DELETE t1 FROM source.tweets t1
INNER JOIN source.tweets t2
WHERE
    t1.id > t2.id AND
    t1.cleansed = t2.cleansed;