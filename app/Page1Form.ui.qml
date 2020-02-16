import QtQuick 2.12
import QtQuick.Controls 2.5

Page {
    id: page
    width: 600
    height: 400

    header: Label {
        text: qsTr("Page 1")
        font.pixelSize: Qt.application.font.pixelSize * 2
        padding: 10
    }

    Button {
        id: button
        x: 250
        y: 182
        text: qsTr("Button")
    }

    ProgressBar {
        id: progressBar
        x: 200
        y: 82
        visible: false
        value: 0.5
    }


    Connections {
        target: button
        onClicked: { page.state = "State1" }
    }

    Image {
        id: image
        x: 250
        y: 40
        width: 100
        height: 100
        fillMode: Image.PreserveAspectFit
        source: "qrc:/qtquickplugin/images/template_image.png"
    }
}
