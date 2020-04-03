
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for TaskGraphPropertyType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="TaskGraphPropertyType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="ProcessNetwork"/>
 *     &lt;enumeration value="DataFlow"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum TaskGraphPropertyType {

    @XmlEnumValue("DataFlow")
    DATA_FLOW("DataFlow"),
    @XmlEnumValue("ProcessNetwork")
    PROCESS_NETWORK("ProcessNetwork");
    private final String value;

    TaskGraphPropertyType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static TaskGraphPropertyType fromValue(String v) {
        for (TaskGraphPropertyType c: TaskGraphPropertyType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
