import streamlit as st
import numpy as np
import pickle


def main():
    st.balloons()
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            crim = st.number_input("CRIM")
            zn = st.number_input("ZN")
            indus = st.number_input("INDUS")
            chas = st.number_input("CHAS")
            nox = st.number_input("NOX")
            rm = st.number_input("RM")
            age = st.number_input("AGE")
        with col2:
            di = st.number_input("DIS")
            rad = st.number_input("RAD")
            tax = st.number_input("TAX")
            ptratio = st.number_input("PTRATIO")
            b = st.number_input("B")
            lstat = st.number_input("LSTAT")

    arr = np.array([
        crim,
        zn,
        indus,
        chas,
        nox,
        rm,
        age,
        di,
        rad,
        tax,
        ptratio,
        b,
        lstat,
    ])

    # st.write(arr)
    if st.button("PREDICT"):
        pickle_model = pickle.load(
            open('boston_house_price_prediction.pkl', 'rb'))
        predicted_val = pickle_model.predict(arr.reshape(1, -1))
        st.header("Boston House Price According to our Model is : " +
                  str(predicted_val[0]))

if __name__ == '__main__':
    main()