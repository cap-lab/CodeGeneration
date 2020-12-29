
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for NetworkType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="NetworkType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="Ethernet/Wi-Fi"/&gt;
 *     &lt;enumeration value="Bluetooth"/&gt;
 *     &lt;enumeration value="USB"/&gt;
 *     &lt;enumeration value="Wire"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "NetworkType")
@XmlEnum
public enum NetworkType {

    @XmlEnumValue("Ethernet/Wi-Fi")
    ETHERNET_WI_FI("Ethernet/Wi-Fi"),
    @XmlEnumValue("Bluetooth")
    BLUETOOTH("Bluetooth"),
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
        throw new IllegalArgumentException(v);
    }

}
