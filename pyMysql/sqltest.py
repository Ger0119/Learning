import pymysql
from pymysql.cursors import DictCursor


def main():
    con = pymysql.connect(host='localhost',port=3306,
                          database='school',
                          user='root',password='root')
    try:
        with con.cursor(cursor=DictCursor) as cursor:
            # name = '立教大学'
            # intro = '日本私立大学'
            # comment = '池袋'
            # update = cursor.execute("insert into college (collname,collintro,comment) values (%s,%s,%s)",(name,intro,comment))
            # if update == 1:
            #     print('Update Success')
            # con.commit()
            cursor.execute('select * from college')
            result = cursor.fetchall()
            print(result)
            for data in result :
                print(data['collid'],end='\t')
                print(data['collname'],end='\t')
                print(data['collintro'])
    finally:
        con.close()


if __name__ == '__main__':
    main()
