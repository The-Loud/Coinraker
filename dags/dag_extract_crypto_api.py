from datetime import timedelta, datetime
from airflow import DAG
from operators.operator_tweet_dump import TweetToMySql
from operators.operator_coin_api import ApiToMySql
from operators.operator_tweet_sentiment import TweetSentiment
from airflow.operators.dummy import DummyOperator
from airflow.providers.mysql.operators.mysql import MySqlOperator

coins = ['bitcoin', 'litecoin', 'ethereum', 'dogecoin']

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['motorific@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

with DAG(
    'coinraker_load_raw',
    default_args=default_args,
    description='Pulls various crypto prices every interval',
    schedule_interval='@hourly',
    start_date=(datetime(2021, 5, 29)),
    catchup=False,
    tags=['crypto']
) as dag:
    t1 = DummyOperator(
        task_id='dummy-1'
    )

    t2 = ApiToMySql(
        task_id='load_stonks',
        name='crypto_task',
        coins=coins,
        method='get_price',
        mysql_conn_id='mysql_pinwheel_source',
        tablename='stonks'
    )

    t3 = ApiToMySql(
        task_id='load_trends',
        name='trends_task',
        mysql_conn_id='mysql_pinwheel_source',
        tablename='trends',
        method='get_search_trending'
    )

    t4 = TweetToMySql(
        task_id='load_tweets',
        name='tweets_task',
        mysql_conn_id='mysql_pinwheel_source',
        tablename='tweets',
        search_query='bitcoin',
        item_count=1000
    )

    t5 = TweetSentiment(
        task_id='calc_sentiment',
        name='sentiment_task',
        mysql_conn_id='mysql_pinwheel_source',
        tablename='sentiment'
    )

    t6 = mysql_task = MySqlOperator(
        task_id='remove_duplicate_tweets',
        mysql_conn_id='mysql_pinwheel_source',
        sql='../sqls/remove_dupes.sql',
    )

    t1 >> [t2, t3, t4]
    t4 >> t6 >> t5
