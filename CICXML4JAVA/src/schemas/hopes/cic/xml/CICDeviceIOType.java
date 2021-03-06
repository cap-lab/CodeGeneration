
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CICDeviceIOType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CICDeviceIOType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="sensors" type="{http://peace.snu.ac.kr/CICXMLSchema}SensorListType" minOccurs="0"/&gt;
 *         &lt;element name="actuators" type="{http://peace.snu.ac.kr/CICXMLSchema}ActuatorListType" minOccurs="0"/&gt;
 *         &lt;element name="displays" type="{http://peace.snu.ac.kr/CICXMLSchema}DisplayListType" minOccurs="0"/&gt;
 *       &lt;/sequence&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CICDeviceIOType", propOrder = {
    "sensors",
    "actuators",
    "displays"
})
public class CICDeviceIOType {

    protected SensorListType sensors;
    protected ActuatorListType actuators;
    protected DisplayListType displays;

    /**
     * Gets the value of the sensors property.
     * 
     * @return
     *     possible object is
     *     {@link SensorListType }
     *     
     */
    public SensorListType getSensors() {
        return sensors;
    }

    /**
     * Sets the value of the sensors property.
     * 
     * @param value
     *     allowed object is
     *     {@link SensorListType }
     *     
     */
    public void setSensors(SensorListType value) {
        this.sensors = value;
    }

    /**
     * Gets the value of the actuators property.
     * 
     * @return
     *     possible object is
     *     {@link ActuatorListType }
     *     
     */
    public ActuatorListType getActuators() {
        return actuators;
    }

    /**
     * Sets the value of the actuators property.
     * 
     * @param value
     *     allowed object is
     *     {@link ActuatorListType }
     *     
     */
    public void setActuators(ActuatorListType value) {
        this.actuators = value;
    }

    /**
     * Gets the value of the displays property.
     * 
     * @return
     *     possible object is
     *     {@link DisplayListType }
     *     
     */
    public DisplayListType getDisplays() {
        return displays;
    }

    /**
     * Sets the value of the displays property.
     * 
     * @param value
     *     allowed object is
     *     {@link DisplayListType }
     *     
     */
    public void setDisplays(DisplayListType value) {
        this.displays = value;
    }

}
