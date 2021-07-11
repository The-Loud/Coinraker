SELECT id ,
       replace(tw.cleansed, ':', '') as cleansed ,
       load_date
FROM   source.tweets tw
WHERE  load_date between date_format(now(), '%%Y-%%m-%%d %%H:00:00') - interval 1 hour
   and date_format(now(), '%%Y-%%m-%%d %%H:00:00')
   and length(tw.cleansed) > 5;
