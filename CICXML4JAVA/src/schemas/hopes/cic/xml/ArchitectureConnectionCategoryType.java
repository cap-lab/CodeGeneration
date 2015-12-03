
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for ArchitectureConnectionCategoryType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="ArchitectureConnectionCategoryType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="Bluetooth"/>
 *     &lt;enumeration value="WIFI"/>
 *     &lt;enumeration value="I2CBus"/>
 *     &lt;enumeration value="VRep_SharedBus"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum ArchitectureConnectionCategoryType {

    @XmlEnumValue("Bluetooth")
    BLUETOOTH("Bluetooth"),
    @XmlEnumValue("I2CBus")
    I_2_C_BUS("I2CBus"),
    @XmlEnumValue("VRep_SharedBus")
    V_REP_SHARED_BUS("VRep_SharedBus"),
    WIFI("WIFI");
    private final String value;

    ArchitectureConnectionCategoryType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static ArchitectureConnectionCategoryType fromValue(String v) {
        for (ArchitectureConnectionCategoryType c: ArchitectureConnectionCategoryType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
