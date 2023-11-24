import matplotlib
from pyspark.sql import SparkSession,functions
spark=SparkSession.builder.appName('shopping').getOrCreate()
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

customers=spark.read.options(header=True).csv('hdfs://localhost:9000/shopping/customers.csv')
orders=spark.read.options(header=True).csv('hdfs://localhost:9000/shopping/orders.csv')
orders = orders.withColumnRenamed("customer_id", "order_customer_id")
products=spark.read.options(header=True).csv('hdfs://localhost:9000/shopping/products.csv')
sales=spark.read.options(header=True).csv('hdfs://localhost:9000/shopping/sales.csv')
sales = sales.withColumnRenamed("order_id", "sale_order_id").withColumnRenamed("product_id","sale_product_id").withColumnRenamed("quantity","sale_quantity")

join_out=customers.join(orders,customers['customer_id']==orders['order_id'])
join_out=join_out.join(products,orders['order_id']==products['product_id'])
join_out=join_out.join(sales,sales['sales_id']==products['product_id'])
join_out.show()
#########################################
# 1)least sold product in each month of 2021

join_out = join_out.withColumn("month", month(join_out["order_date"]))
table1 = join_out.select("product_id", "product_name", "quantity", "month")
grouped_data = table1.groupBy("month", "product_name").agg(min("quantity").alias("min_quantity"))
grouped_data.show()
window_spec = Window.partitionBy("month").orderBy(col("min_quantity").desc())
result = grouped_data.withColumn("rank", rank().over(window_spec)).filter(col("rank") == 1).drop("rank")
result.show()
# Create a bar diagram
result_pandas = result.toPandas()
for product_name in result_pandas['product_name'].unique():
    data = result_pandas[result_pandas['product_name'] == product_name]
    plt.bar(data['month'], data['min_quantity'], label=product_name)

plt.xlabel('Month')
plt.ylabel('Min Quantity')
plt.title('Minimum Quantity by Product and Month')
plt.legend()
plt.show()

#######################################################
# 2)Total number of delivery in each state

table2=join_out.select("state","delivery_date")
total_delivery = table2.groupBy("state").agg(count("delivery_date").alias("delivery_count"))
total_delivery.show()
# Create a bar diagram
grouped_data_pandas = total_delivery.toPandas()
# Create a bar diagram
plt.bar(grouped_data_pandas['state'], grouped_data_pandas['delivery_count'])
plt.xlabel('state')
plt.ylabel('delivery_count')
plt.title('Delivery Count by State')
plt.show()

######################################################
# 3)What is the most common age group among customers

table3=join_out.select("customer_id","age")
common_age=table3.groupBy("age").agg(count("customer_id").alias("total_customers")).orderBy(desc("total_customers"))
common_age.show()
# Create a bar diagram
common_age_pandas = common_age.toPandas()
plt.bar(common_age_pandas['age'], common_age_pandas['total_customers'])
plt.xlabel('age')
plt.ylabel('total_customers')
plt.title('Customer Count by Age')
plt.show()

#######################################################
# 4)Calculate the average time it takes for orders to be delivered after they were placed and find Which orders had the shortest and longest delivery days

table4 = join_out.select("order_id", "customer_name", "order_date", "delivery_date")
table4 = table4.withColumn("days_between", datediff(col("delivery_date"), col("order_date")))
average_delivery_time = table4.agg(avg("days_between")).first()[0]
print(f"Average Delivery Time: {average_delivery_time:.2f} days")
shortest_delivery = table4.orderBy(asc("days_between")).first()
print("Order with the Shortest Delivery Time:",shortest_delivery)
longest_delivery = table4.orderBy(desc("days_between")).first()
print("Order with the Longest Delivery Time:",longest_delivery)

#####################################################
# 5)Calculate the total sales or revenue for each product type

table5=join_out.select("product_id","product_type","sale_quantity","total_price")
total_sales=table5.groupBy("product_type").agg(sum("total_price").alias("total_revenue"))
total_sales.show()
# Create a bar diagram
product_types = total_sales.select("product_type").rdd.flatMap(lambda x: x).collect()
total_revenues = total_sales.select("total_revenue").rdd.flatMap(lambda x: x).collect()
plt.bar(product_types, total_revenues, color='blue', edgecolor='black')
plt.xlabel('Product Type')
plt.ylabel('Total Revenue')
plt.title('Total Revenue per Product Type')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()

######################################################
# 6)Find products with low inventory levels that need to be restocked,threshold value less than 40 need to be restocked

table6=join_out.select("product_id","quantity","sale_quantity")
low_inventory=table6.withColumn("quantity_difference",col("quantity")-col("sale_quantity")).orderBy(asc("quantity_difference"))
low_inventory.show()
threshold_value=40
low_inventory_products = low_inventory.filter(col("quantity_difference") < threshold_value)
low_inventory_products.select("product_ID","quantity","sale_quantity","quantity_difference").show()
count=low_inventory_products.count()
print(count)
# Create a bar diagram
product_ids = low_inventory_products.select("product_id").rdd.flatMap(lambda x: x).collect()
quantity_differences = low_inventory_products.select("quantity_difference").rdd.flatMap(lambda x: x).collect()
plt.bar(product_ids, quantity_differences, color='red', edgecolor='black')
plt.xlabel('Product ID')
plt.ylabel('Quantity Difference')
plt.title(f'Products with Quantity Difference Below {threshold_value}')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()

#############################################
# 7)Determine the busiest shopping days of the week.

table7=join_out.select("order_id","order_date")
days= table7.withColumn("day_of_week", date_format(col("order_date"), "EEEE"))
busiest_day = days.groupBy("day_of_week").agg(count("order_id").alias("no_of_orders")).orderBy(desc("no_of_orders"))
busiest_day.show()
# Create a bar diagram
pandas_df = busiest_day.toPandas()
# Plotting using Matplotlib
plt.bar(pandas_df["day_of_week"], pandas_df["no_of_orders"], color='green')
plt.xlabel("Day of the Week")
plt.ylabel("Number of Orders")
plt.title("Busiest Day by Number of Orders")
plt.show()

##############################################
# 8)Show the products available in a specific size or color.

table8 = join_out.select("product_ID", "product_name", "size", "colour")
df1 = table8.groupBy("size", "colour").agg(count("product_id").alias("no_of_products"))
specific_size = input("enter the size:-").upper()
specific_color = input("enter the colour:-").lower()
filtered_products = df1.filter((df1["size"] == specific_size) & (df1["colour"] == specific_color))
filtered_products.show()

############################################################
# 9)Identify customers who have made highest purchase amount.

table9=join_out.select("customer_name","sale_order_id","total_price","product_id","product_name")
result=table9.groupBy("customer_name","product_id","product_name").agg(sum("total_price").alias("price")).orderBy(desc("price"))
result.show()
# Create a bar diagram
pandas_df = result.toPandas()
top_10_customers = pandas_df.groupby("customer_name")["price"].sum().nlargest(10)
# Plotting using Matplotlib as a bar graph
plt.figure(figsize=(10, 6))
top_10_customers.plot(kind='bar', color='purple')
plt.xlabel("Customer Name")
plt.ylabel("Total Price")
plt.title("Top 10 Customers by Total Price")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

##########################################################
# 10)Analyze whether there are seasonal trends in product sale.

table10=join_out.select("sales_id","product_name","order_date")
sales_df = table10.withColumn("month", month("order_date"))
sales_df = sales_df.withColumn("quarter", quarter("order_date"))
sales_df.show()
monthly_trends = sales_df.groupBy("month").count().orderBy("month")
monthly_trends.show()
# Create a bar diagram
monthly_trends_pd = monthly_trends.toPandas()
plt.figure(figsize=(10, 5))
plt.bar(monthly_trends_pd["month"], monthly_trends_pd["count"])
plt.xlabel("Month")
plt.ylabel("Number of Sales")
plt.title("Monthly Sales Trends")
plt.show()

#############################################################
