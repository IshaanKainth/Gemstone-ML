#!/bin/sh
airflow scheduler &  # Start the scheduler in background
airflow webserver    # Start the webserver
