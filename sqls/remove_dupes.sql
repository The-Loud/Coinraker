insert into dd_tweets
with  cte as (
select	*,
		ROW_NUMBER() OVER (PARTITION BY cleansed order by cleansed) row_num
from tweets t
where t.load_date between date_format(now(), '%Y-%m-%d %H:00:00') - interval 1 hour
                                             and date_format(now(), '%Y-%m-%d %H:00:00'))
SELECT created_at, id, `text`, entities, lang, load_date, cleaned_text, cleansed
from cte
where row_num = 1
