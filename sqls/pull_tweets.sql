select text
from tweets
where HOUR(load_date) between HOUR(NOW()) and HOUR(NOW() - 1)