from datetime import timedelta, datetime
from airflow import DAG
from operators import ApiToMySql, TweetToMySql
from airflow.operators.dummy import DummyOperator

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
    'crypton_dag_trends',
    default_args=default_args,
    description='Pulls various crypto prices every interval',
    schedule_interval='@hourly',
    start_date=(datetime(2021, 5, 15)),
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
        item_count=100
    )

    t1 >> [t2, t3, t4]
