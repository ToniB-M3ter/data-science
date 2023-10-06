import sys, importlib
import exponential_smooth as es

dataloadcache = None
if __name__ == "__main__":
    while True:
        if not dataloadcache:
            dataloadcache = es.get_data()
        es.part2(dataloadcache)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        importlib.reload(es)