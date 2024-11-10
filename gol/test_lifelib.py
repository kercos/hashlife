import lifelib

def test():
    sess = lifelib.load_rules("b3s23")
    lt = sess.lifetree()
    lidka = lt.pattern("bo7b$obo6b$bo7b8$8bo$6bobo$5b2obo2$4b3o!")
    print("Initial population: %d" % lidka.population)
    lidka_30k = lidka[30000]
    print("Final population: %d" % lidka_30k.population)
    lidka.viewer()


if __name__ == "__main__":
    test()