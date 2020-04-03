
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for NetworkType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="NetworkType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="Ethernet/Wi-Fi"/>
 *     &lt;enumeration value="Bluetooth"/>
 *     &lt;enumeration value="USB"/>
 *     &lt;enumeration value="Wire"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum NetworkType {

    @XmlEnumValue("Bluetooth")
    BLUETOOTH("Bluetooth"),
    @XmlEnumValue("Ethernet/Wi-Fi")
    ETHERNET_WI_FI("Ethernet/Wi-Fi"),
    USB("USB"),
    @XmlEnumValue("Wire")
    WIRE("Wire");
    private final String value;

    NetworkType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static NetworkType fromValue(String v) {
        for (NetworkType c: NetworkType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
