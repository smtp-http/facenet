import socket
import threading
import time
import sys
import os
import struct


class __tcp_server:
    def socket_service():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', 6666))
            s.listen(10)
        except socket.error as msg:
            print(msg)
            sys.exit(1)
        print('Waiting connection...')

        while 1:
            conn, addr = s.accept()
            t = threading.Thread(target=deal_data, args=(conn, addr))
            t.start()


    def deal_data(conn, addr): 
        print ('Accept new connection from {0}'.format(addr))
        #conn.settimeout(500) 
        conn.send(('Hi, Welcome to the server!').encode())

        while 1:
            fileinfo_size = struct.calcsize('128sl')
            buf = conn.recv(fileinfo_size)
            if buf:
            	filename, filesize = struct.unpack('128sl', buf) 
            	fn = filename.strip(('\00').encode())
            	#src_path,_ = os.path.split(os.path.realpath(__file__))
            	new_filename = os.path.join('./', 'new_' + fn.decode())
            	print('file new name is {0}, filesize if {1}'.format(new_filename, filesize))
            	recvd_size = 0 # 定义已接收文件的大小 
            	fp = open(new_filename, 'wb') 
            	print('start receiving...')

            	while not recvd_size == filesize: 
            		if filesize - recvd_size > 1024: 
            			data = conn.recv(1024) 
            			recvd_size += len(data) 
            		else: 
            			data = conn.recv(filesize - recvd_size) 
            			recvd_size = filesize 
            		fp.write(data) 
            	fp.close() 
            	print('end receive...')
            conn.close() 
            break

class __tcp_client:
    def socket_client():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('127.0.0.1', 6666))
        except socket.error as msg:
            print(msg)
            sys.exit(1)

        print(s.recv(1024))

        while 1: 
            filepath = input('please input file path: ') 
            if os.path.isfile(filepath): 
            # 定义定义文件信息。128s表示文件名为128bytes长，l表示一个int或log文件类型，在此为文件大小
                fileinfo_size = struct.calcsize('128sl') 
            # 定义文件头信息，包含文件名和文件大小 
                fhead = struct.pack(('128sl').encode(), os.path.basename(filepath.encode()), os.stat(filepath).st_size)
                s.send(fhead)
                print('client filepath: {0}'.format(filepath))

                fp = open(filepath, 'rb')
                while 1:
                    data = fp.read(1024)
                    if not data:
                        print('{0} file send over...'.format(filepath))
                        break
                    s.send(data)
            s.close()
            break