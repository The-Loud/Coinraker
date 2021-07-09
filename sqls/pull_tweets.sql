select id, REPLACE(tw.cleansed, ':', '') AS `cleansed`
from source.tweets tw
where load_date between DATE_FORMAT(NOW(), '%Y-%m-%d %H:00:00') - INTERVAL 1 HOUR and DATE_FORMAT(NOW(), '%Y-%m-%d %H:00:00')
and LENGTH(tw.cleansed) > 5
