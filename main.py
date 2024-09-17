import cv2
from services.draw_results import draw_results
from services.process_frame import process_frame
from services.load_known_faces import (
    load_known_faces,
)


def main():

    directory = "./images"
    # Carrega as codificações faciais e os nomes a partir dos diretórios
    known_face_encodings, known_face_names = load_known_faces(directory)

    # ========================================
    video_capture = cv2.VideoCapture(0)
    # ========================================
    # Modulariza isso ai
    process_this_frame = True
    face_locations, face_names = [], []
    # ================================================================
    # ================================================================

    while True:
        ret, frame = video_capture.read()

        # ================================================================
        # ================================================================
        # Modulariza isso ai
        # Processa frames alternados para melhorar o desempenho
        if process_this_frame:
            face_locations, face_names = process_frame(
                frame, known_face_encodings, known_face_names
            )
        # ================================================================
        # ================================================================

        # ================================================================
        # Desenha os resultados no frame
        frame = draw_results(frame, face_locations, face_names)

        # Alterna o processamento de frames
        process_this_frame = not process_this_frame

        # Exibe o frame resultante
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
